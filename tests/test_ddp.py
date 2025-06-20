#!/usr/bin/env python3
"""
Simple DDP test script to verify implementation.
"""

import os
import torch
import hydra
from omegaconf import DictConfig

# Import our training functions
from train import setup_ddp, cleanup_ddp, is_main_process
from model import create_model
from dataloader import create_dataloader, InfiniteDataLoader, get_batch


@hydra.main(version_base=None, config_path=".", config_name="config")
def test_ddp(cfg: DictConfig) -> None:
    """Test DDP setup with a few training steps."""
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Set device
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42 + rank)
    
    if is_main_process():
        print(f"Testing DDP setup...")
        print(f"World size: {world_size}")
        print(f"Using device: {device}")
    
    # Create a small model for testing
    vocab_size = 50257  # Dummy vocab size
    model = create_model(vocab_size)
    model.to(device)
    
    if is_main_process():
        print(f"Model parameters: {model.get_num_params():,}")
    
    # Wrap with DDP if multi-GPU
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        if is_main_process():
            print(f"Model wrapped with DDP")
    
    # Create data loader (reduced batch size for testing)
    train_loader, actual_vocab_size = create_dataloader(
        cfg.data.data_path,
        batch_size=4,  # Small batch for testing
        seq_len=256,   # Shorter sequence for testing
        train=True,
        num_workers=2,
    )
    
    train_loader = InfiniteDataLoader(train_loader)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    if is_main_process():
        print("Starting test training...")
    
    # Test a few training steps
    model.train()
    for step in range(5):
        # Forward pass
        x, y = get_batch(train_loader, device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if is_main_process():
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    if is_main_process():
        print("DDP test completed successfully!")
    
    # Clean up
    cleanup_ddp()


if __name__ == "__main__":
    test_ddp() 