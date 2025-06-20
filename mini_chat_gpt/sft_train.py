"""
Supervised Fine-Tuning (SFT) script for instruction following.
Adapted from train.py with loss masking for chat format.
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
import yaml
try:
    import wandb
except ImportError:
    wandb = None

from mini_chat_gpt.model import create_model, GPT
from mini_chat_gpt.sft_dataloader import create_sft_dataloader


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


def estimate_sft_loss(model: GPT, val_loader, device: str, eval_iters: int = 50) -> float:
    """Estimate validation loss for SFT."""
    model.eval()
    losses = []
    
    for _ in range(eval_iters):
        try:
            input_ids, labels = next(iter(val_loader))
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(input_ids)
                    
                    # Compute loss with masking
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), 
                        labels.view(-1), 
                        ignore_index=-100,
                        reduction='mean'
                    )
                    
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    losses.append(loss.item())
        except StopIteration:
            break
    
    model.train()
    return sum(losses) / len(losses) if losses else float('inf')


def save_sft_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, step: int, loss: float, save_dir: str) -> None:
    """Save SFT model checkpoint."""
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
        },
        'model_type': 'sft'  # Mark as SFT checkpoint
    }
    
    save_path = os.path.join(save_dir, f"sft_checkpoint_step_{step}.pt")
    torch.save(checkpoint, save_path)
    
    if is_main_process():
        print(f"SFT checkpoint saved to {save_path}")


def load_pretrained_checkpoint(checkpoint_path: str, model: GPT) -> None:
    """Load pretrained checkpoint into model."""
    if is_main_process():
        print(f"Loading pretrained checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle DDP model - get the underlying module
    model_to_load = model.module if hasattr(model, 'module') else model
    
    # Handle state dict keys that may have _orig_mod. prefix from torch.compile
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Remove _orig_mod. prefix if present (from torch.compile)
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model_to_load.load_state_dict(new_state_dict)
    
    if is_main_process():
        print("Pretrained checkpoint loaded successfully")


def load_sft_checkpoint(checkpoint_path: str, model: GPT, optimizer: torch.optim.Optimizer) -> int:
    """Load SFT checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle DDP model - get the underlying module
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    
    if is_main_process():
        print(f"SFT checkpoint loaded from step {step}")
    return step


def main(config_path: str = "sft_config.yaml") -> None:
    """Main SFT training function."""
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace for easier access
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    
    cfg = Config(config_dict)
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Set device and seed  
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    seed = getattr(cfg.system, 'seed', 42) if hasattr(cfg, 'system') else 42
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    
    # Fix for torch.compile + DDP compatibility
    compile_enabled = getattr(cfg.system, 'compile', False) if hasattr(cfg, 'system') else False
    if compile_enabled and world_size > 1:
        torch._dynamo.config.optimize_ddp = False
        if is_main_process():
            print("Disabled DDP optimizer for torch.compile compatibility")
    
    if is_main_process():
        print(f"Starting SFT training...")
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
        print(f"Config loaded from: {config_path}")
    
    # Initialize WandB (only on main process)
    wandb_enabled = getattr(cfg.wandb, 'enabled', False) if hasattr(cfg, 'wandb') else False
    if wandb_enabled and is_main_process() and wandb is not None:
        wandb.init(
            project=getattr(cfg.wandb, 'project', 'sft_training'),
            name=getattr(cfg.wandb, 'name', 'sft_run'),
            config=config_dict
        )
    
    # Create SFT data loaders  
    dataset_path = cfg.sft.dataset_path
    batch_size = getattr(cfg.training, 'batch_size', 4)
    seq_len = getattr(cfg.training, 'seq_len', 512)
    num_workers = getattr(cfg.training, 'num_workers', 0)
    
    train_loader, vocab_size = create_sft_dataloader(
        dataset_path,
        batch_size,
        seq_len,
        train=True,
        num_workers=num_workers,
    )
    
    val_loader, _ = create_sft_dataloader(
        dataset_path,
        batch_size,
        seq_len,
        train=False,
        num_workers=num_workers,
    )
    
    if is_main_process():
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(vocab_size)
    model.to(device)
    
    if is_main_process():
        print(f"Model parameters: {model.get_num_params():,}")
    
    # Load pretrained checkpoint
    pretrained_checkpoint = getattr(cfg.sft, 'pretrained_checkpoint', None)
    if pretrained_checkpoint:
        load_pretrained_checkpoint(pretrained_checkpoint, model)
    else:
        if is_main_process():
            print("Warning: No pretrained checkpoint specified. Training from scratch.")
    
    # Compile model if requested (before DDP wrapping)
    if compile_enabled and hasattr(torch, 'compile'):
        if is_main_process():
            print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Multi-GPU support with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        if is_main_process():
            print(f"Using DDP with {world_size} GPUs")
            effective_batch_size = batch_size * world_size
            print(f"Effective batch size: {effective_batch_size}")
    else:
        if is_main_process():
            print("Using single GPU")
    
    # Create optimizer with lower learning rate for SFT
    learning_rate = getattr(cfg.sft, 'learning_rate', 2e-5)
    beta1 = getattr(cfg.training, 'beta1', 0.9) if hasattr(cfg, 'training') else 0.9
    beta2 = getattr(cfg.training, 'beta2', 0.999) if hasattr(cfg, 'training') else 0.999
    weight_decay = getattr(cfg.training, 'weight_decay', 0.1) if hasattr(cfg, 'training') else 0.1
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    
    # Create GradScaler for mixed precision
    mixed_precision = getattr(cfg.system, 'mixed_precision', False) if hasattr(cfg, 'system') else False
    scaler = torch.amp.GradScaler('cuda') if mixed_precision else None
    
    # Resume from SFT checkpoint if specified
    start_step = 0
    resume_from = getattr(cfg.sft, 'resume_from', None)
    if resume_from:
        start_step = load_sft_checkpoint(resume_from, model, optimizer)
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    
    # Get training parameters
    max_steps = getattr(cfg.sft, 'max_steps', 10000)
    warmup_steps = getattr(cfg.sft, 'warmup_steps', 500)
    eval_interval = getattr(cfg.sft, 'eval_interval', 500)
    save_interval = getattr(cfg.sft, 'save_interval', 1000)
    save_dir = getattr(cfg.sft, 'save_dir', 'checkpoints')
    grad_clip = getattr(cfg.training, 'grad_clip', 1.0) if hasattr(cfg, 'training') else 1.0
    
    # Calculate tokens per batch for throughput calculation
    tokens_per_batch = batch_size * seq_len
    if world_size > 1:
        tokens_per_batch *= world_size
    
    # For tracking average throughput
    tokens_processed = 0
    time_elapsed = 0.0
    
    # Training data iterator
    train_iter = iter(train_loader)
    
    for step in range(start_step, max_steps):
        t0 = time.time()
        
        # Get learning rate
        lr = get_lr(step, warmup_steps, max_steps, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        try:
            input_ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, labels = next(train_iter)
        
        input_ids, labels = input_ids.to(device), labels.to(device)
        
        # Mixed precision context
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if mixed_precision else nullcontext()
        
        with autocast_ctx:
            logits, _ = model(input_ids)
            
            # Compute loss with masking (only train on assistant tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100,
                reduction='mean'
            )
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
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
        if step % 50 == 0 and is_main_process():
            print(f"SFT Step {step:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms | Tokens/sec: {tokens_per_second:.0f}")
        
        # Evaluation (only on main process)
        if step % eval_interval == 0 and step > 0 and is_main_process():
            val_loss = estimate_sft_loss(model, val_loader, device)
            print(f"SFT Step {step:6d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            
            if wandb_enabled and wandb is not None:
                wandb.log({
                    "sft/train_loss": loss.item(),
                    "sft/val_loss": val_loss,
                    "sft/lr": lr,
                    "sft/tokens_per_second": tokens_per_second,
                    "step": step,
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_sft_checkpoint(model, optimizer, step, val_loss, save_dir)
        
        # Regular checkpointing (only on main process)
        if step % save_interval == 0 and step > 0 and is_main_process():
            save_sft_checkpoint(model, optimizer, step, loss.item(), save_dir)
        
        # Log to WandB (only on main process)
        if wandb_enabled and step % 50 == 0 and is_main_process() and wandb is not None:
            wandb.log({
                "sft/train_loss": loss.item(),
                "sft/lr": lr,
                "sft/tokens_per_second": tokens_per_second,
                "step": step,
            })
    
    # Final checkpoint (only on main process)
    if is_main_process():
        save_sft_checkpoint(model, optimizer, max_steps, loss.item(), save_dir)
        print("SFT training completed!")
    
    # Clean up DDP
    cleanup_ddp()


if __name__ == "__main__":
    main() 