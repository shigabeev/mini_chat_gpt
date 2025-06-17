"""
Training script for GPT on TinyStories.
Features: Mixed precision, gradient accumulation, WandB logging.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from model import create_model, GPT
from dataloader import create_dataloader, InfiniteDataLoader, get_batch


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    if step > max_steps:
        return 0.0
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * coeff


def estimate_loss(model: GPT, val_loader, device: str, eval_iters: int = 100) -> float:
    """Estimate validation loss."""
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        with torch.no_grad():
            x, y = get_batch(val_loader, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, step: int, loss: float, save_dir: str) -> None:
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle DataParallel model
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss,
        'config': {
            'vocab_size': model_to_save.vocab_size,
            'dim': model_to_save.dim,
            'n_layers': len(model_to_save.blocks),
            'n_heads': model_to_save.blocks[0].attn.n_heads,
            'max_seq_len': model_to_save.max_seq_len,
        }
    }
    
    save_path = os.path.join(save_dir, f"checkpoint_step_{step}.pt")
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path: str, model: GPT, optimizer: torch.optim.Optimizer) -> int:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    print(f"Loaded checkpoint from step {step}")
    return step


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Set device and seed
    device = cfg.system.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.system.seed)
    torch.cuda.manual_seed(cfg.system.seed)
    
    print(f"Using device: {device}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize WandB
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Create data loaders
    train_loader, vocab_size = create_dataloader(
        cfg.data.data_path,
        cfg.training.batch_size,
        cfg.data.seq_len,
        train=True,
        num_workers=cfg.data.num_workers,
    )
    
    val_loader, _ = create_dataloader(
        cfg.data.data_path,
        cfg.training.batch_size,
        cfg.data.seq_len,
        train=False,
        num_workers=cfg.data.num_workers,
    )
    
    # Wrap in infinite loader
    train_loader = InfiniteDataLoader(train_loader)
    val_loader = InfiniteDataLoader(val_loader)
    
    # Create model
    model = create_model(vocab_size)
    model.to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Compile model if requested (before DataParallel wrapping)
    if cfg.system.compile and hasattr(torch, 'compile'):
        if cfg.system.multi_gpu and torch.cuda.device_count() > 1:
            print("Warning: torch.compile with DataParallel can be unstable. Consider using single GPU or DistributedDataParallel.")
            print("Disabling compilation for multi-GPU training.")
        else:
            print("Compiling model...")
            model = torch.compile(model)
    
    # Multi-GPU support
    if cfg.system.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        # Adjust batch size for multiple GPUs
        effective_batch_size = cfg.training.batch_size * torch.cuda.device_count()
        print(f"Effective batch size: {effective_batch_size}")
    else:
        print("Using single GPU")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        weight_decay=cfg.training.weight_decay,
    )
    
    # Create GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda') if cfg.system.mixed_precision else None
    
    # Resume from checkpoint if specified
    start_step = 0
    if cfg.checkpointing.resume_from:
        start_step = load_checkpoint(cfg.checkpointing.resume_from, model, optimizer)
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    # Calculate tokens per batch for throughput calculation
    tokens_per_batch = cfg.training.batch_size * cfg.data.seq_len
    if cfg.system.multi_gpu and torch.cuda.device_count() > 1:
        tokens_per_batch *= torch.cuda.device_count()
    
    # For tracking average throughput
    tokens_processed = 0
    time_elapsed = 0.0
    
    for step in range(start_step, cfg.training.max_steps):
        t0 = time.time()
        
        # Get learning rate
        lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps, cfg.training.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        x, y = get_batch(train_loader, device)
        
        # Mixed precision context
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if cfg.system.mixed_precision else nullcontext()
        
        with autocast_ctx:
            logits, loss = model(x, y)
        
        # Ensure loss is scalar for DataParallel
        if loss.dim() > 0:
            loss = loss.mean()
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        t1 = time.time()
        dt = t1 - t0
        
        # Calculate tokens per second
        tokens_per_second = tokens_per_batch / dt
        
        # Update running averages
        tokens_processed += tokens_per_batch
        time_elapsed += dt
        avg_tokens_per_second = tokens_processed / time_elapsed if time_elapsed > 0 else 0
        
        # Logging
        if step % 100 == 0:
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms | Tokens/sec: {tokens_per_second:.0f} (avg: {avg_tokens_per_second:.0f})")
        
        # Evaluation
        if step % cfg.training.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(model, val_loader, device)
            print(f"Step {step:6d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Tokens/sec: {tokens_per_second:.0f} (avg: {avg_tokens_per_second:.0f})")
            
            if cfg.wandb.enabled:
                wandb.log({
                    "train/loss": loss.item(),
                    "val/loss": val_loss,
                    "train/lr": lr,
                    "train/tokens_per_second": tokens_per_second,
                    "train/avg_tokens_per_second": avg_tokens_per_second,
                    "step": step,
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss, cfg.checkpointing.save_dir)
        
        # Regular checkpointing
        if step % cfg.training.save_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, loss.item(), cfg.checkpointing.save_dir)
        
        # Log to WandB
        if cfg.wandb.enabled and step % 100 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/tokens_per_second": tokens_per_second,
                "train/avg_tokens_per_second": avg_tokens_per_second,
                "step": step,
            })
    
    # Final checkpoint
    save_checkpoint(model, optimizer, cfg.training.max_steps, loss.item(), cfg.checkpointing.save_dir)
    print("Training completed!")


if __name__ == "__main__":
    main() 
    