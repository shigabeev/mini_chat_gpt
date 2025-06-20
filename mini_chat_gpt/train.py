"""
Training script for GPT on TinyStories.
Features: Mixed precision, gradient accumulation, WandB logging, DDP support.
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch._dynamo
from contextlib import nullcontext
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from mini_chat_gpt.model import create_model, GPT
from mini_chat_gpt.dataloader import create_dataloader, InfiniteDataLoader, get_batch


def setup_ddp():
    """Initialize DDP. Call this before creating model."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU or CPU training
        return 0, 1, 0


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


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
                # Ensure loss is scalar for DataParallel
            if loss.dim() > 0:
                loss = loss.mean()
            losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, step: int, loss: float, save_dir: str) -> None:
    """Save model checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle DDP model - get the underlying module
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
    
    # Handle DDP model - get the underlying module
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    
    if is_main_process():
        print(f"Loaded checkpoint from step {step}")
    return step


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Set device and seed
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.system.seed + rank)  # Different seed per rank
    torch.cuda.manual_seed(cfg.system.seed + rank)
    
    # Fix for torch.compile + DDP compatibility
    if cfg.system.compile and world_size > 1:
        torch._dynamo.config.optimize_ddp = False
        if is_main_process():
            print("Disabled DDP optimizer for torch.compile compatibility")
    
    if is_main_process():
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
        print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize WandB (only on main process)
    if cfg.wandb.enabled and is_main_process():
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
    
    if is_main_process():
        print(f"Model parameters: {model.get_num_params():,}")
    
    # Compile model if requested (before DDP wrapping)
    if cfg.system.compile and hasattr(torch, 'compile'):
        if is_main_process():
            print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Multi-GPU support with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if is_main_process():
            print(f"Using DDP with {world_size} GPUs")
            effective_batch_size = cfg.training.batch_size * world_size
            print(f"Effective batch size: {effective_batch_size}")
    else:
        if is_main_process():
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
    if world_size > 1:
        tokens_per_batch *= world_size
    
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
        
        # DDP already handles loss averaging across processes
        
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
        
        # Logging (only on main process)
        if step % 100 == 0 and is_main_process():
            print(f"Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms | Tokens/sec: {tokens_per_second:.0f} (avg: {avg_tokens_per_second:.0f})")
        
        # Evaluation (only on main process)
        if step % cfg.training.eval_interval == 0 and step > 0 and is_main_process():
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
        
        # Regular checkpointing (only on main process)
        if step % cfg.training.save_interval == 0 and step > 0 and is_main_process():
            save_checkpoint(model, optimizer, step, loss.item(), cfg.checkpointing.save_dir)
        
        # Log to WandB (only on main process)
        if cfg.wandb.enabled and step % 100 == 0 and is_main_process():
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/tokens_per_second": tokens_per_second,
                "train/avg_tokens_per_second": avg_tokens_per_second,
                "step": step,
            })
    
    # Final checkpoint (only on main process)
    if is_main_process():
        save_checkpoint(model, optimizer, cfg.training.max_steps, loss.item(), cfg.checkpointing.save_dir)
        print("Training completed!")
    
    # Clean up DDP
    cleanup_ddp()


if __name__ == "__main__":
    main() 
    