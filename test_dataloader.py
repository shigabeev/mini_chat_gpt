"""
Comprehensive tests for the TinyStories dataloader.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, mock_open
from dataloader import TinyStoriesDataset, create_dataloader, InfiniteDataLoader, get_batch


class TestTinyStoriesDataset:
    """Test the TinyStoriesDataset class."""
    
    def test_dataset_initialization(self, tmp_path):
        """Test basic dataset initialization."""
        # Create mock data
        mock_text = "Once upon a time there was a little girl.\n\nShe loved to play with toys.\n\nThe end."
        
        # Create mock tokenized data
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 100, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataset = TinyStoriesDataset(str(tmp_path / "data"), seq_len=32, train=True)
        
        assert dataset.seq_len == 32
        assert dataset.train == True
        assert len(dataset.data) == 1000
        assert dataset.vocab_size > 0
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset item retrieval."""
        # Create test data
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(100, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataset = TinyStoriesDataset(str(tmp_path / "data"), seq_len=10, train=True)
        
        x, y = dataset[0]
        
        assert x.shape == (10,)
        assert y.shape == (10,)
        assert torch.equal(x, torch.arange(10))
        assert torch.equal(y, torch.arange(1, 11))
    
    def test_dataset_length(self, tmp_path):
        """Test dataset length calculation."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(100, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataset = TinyStoriesDataset(str(tmp_path / "data"), seq_len=10, train=True)
        
        # Length should be total_tokens - seq_len
        assert len(dataset) == 100 - 10
    
    def test_dataset_boundaries(self, tmp_path):
        """Test dataset boundary conditions."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(50, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataset = TinyStoriesDataset(str(tmp_path / "data"), seq_len=10, train=True)
        
        # Test first item
        x, y = dataset[0]
        assert torch.equal(x, torch.arange(10))
        assert torch.equal(y, torch.arange(1, 11))
        
        # Test last item
        last_idx = len(dataset) - 1
        x, y = dataset[last_idx]
        assert torch.equal(x, torch.arange(39, 49))
        assert torch.equal(y, torch.arange(40, 50))
    
    def test_train_vs_val_split(self, tmp_path):
        """Test that train and val datasets load different files."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        
        # Create different data for train and val
        train_tokens = np.arange(100, dtype=np.uint16)
        val_tokens = np.arange(100, 200, dtype=np.uint16)
        
        train_tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        val_tokens.tofile(tmp_path / "data" / "tinystories_val.bin")
        
        train_dataset = TinyStoriesDataset(str(tmp_path / "data"), seq_len=10, train=True)
        val_dataset = TinyStoriesDataset(str(tmp_path / "data"), seq_len=10, train=False)
        
        train_x, _ = train_dataset[0]
        val_x, _ = val_dataset[0]
        
        assert not torch.equal(train_x, val_x)
        assert torch.equal(train_x, torch.arange(10))
        assert torch.equal(val_x, torch.arange(100, 110))


class TestDataLoaderCreation:
    """Test dataloader creation and configuration."""
    
    def test_create_dataloader_basic(self, tmp_path):
        """Test basic dataloader creation."""
        # Setup test data
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataloader, vocab_size = create_dataloader(
            str(tmp_path / "data"),
            batch_size=4,
            seq_len=32,
            train=True,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        assert vocab_size > 0
        assert dataloader.batch_size == 4
        assert dataloader.drop_last == True
        assert dataloader.pin_memory == True
    
    def test_dataloader_batch_shape(self, tmp_path):
        """Test that dataloader produces correct batch shapes."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataloader, _ = create_dataloader(
            str(tmp_path / "data"),
            batch_size=8,
            seq_len=64,
            train=True,
            num_workers=0
        )
        
        x, y = next(iter(dataloader))
        
        assert x.shape == (8, 64)
        assert y.shape == (8, 64)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64
    
    def test_dataloader_shuffle_behavior(self, tmp_path):
        """Test shuffle behavior for train vs val."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        tokens.tofile(tmp_path / "data" / "tinystories_val.bin")
        
        train_loader, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=True, num_workers=0
        )
        val_loader, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=False, num_workers=0
        )
        
        # Check sampler types instead of shuffle attribute
        from torch.utils.data import RandomSampler, SequentialSampler
        assert isinstance(train_loader.sampler, RandomSampler)
        assert isinstance(val_loader.sampler, SequentialSampler)
    
    def test_dataloader_consistency(self, tmp_path):
        """Test that same data produces consistent results."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_val.bin")  # Use val for no shuffle
        
        dataloader1, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=False, num_workers=0
        )
        dataloader2, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=False, num_workers=0
        )
        
        x1, y1 = next(iter(dataloader1))
        x2, y2 = next(iter(dataloader2))
        
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


class TestInfiniteDataLoader:
    """Test the InfiniteDataLoader wrapper."""
    
    def test_infinite_iteration(self, tmp_path):
        """Test that infinite dataloader cycles properly."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(50, dtype=np.uint16)  # Very small dataset for quick cycling
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataloader, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=2, seq_len=8, train=False, num_workers=0
        )
        
        # Should only have a few batches  
        finite_batches = len(dataloader)
        assert finite_batches > 0
        
        infinite_loader = InfiniteDataLoader(dataloader)
        
        # Test just 5 batches to verify cycling works
        batches = []
        for i, batch in enumerate(infinite_loader):
            batches.append(batch)
            if i >= 4:  # Just get 5 batches
                break
        
        assert len(batches) == 5
        
        # Test that we can get batches beyond the finite loader size
        if finite_batches < 5:
            # First and cycled batches should be identical (no shuffle for val)
            x1, y1 = batches[0]
            x_cycled, y_cycled = batches[finite_batches]
            
            assert torch.equal(x1, x_cycled)
            assert torch.equal(y1, y_cycled)
    
    def test_infinite_next_method(self, tmp_path):
        """Test the __next__ method directly."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(100, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_val.bin")
        
        dataloader, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=2, seq_len=10, train=False, num_workers=0
        )
        
        infinite_loader = InfiniteDataLoader(dataloader)
        
        # Test multiple next() calls
        batch1 = next(infinite_loader)
        batch2 = next(infinite_loader)
        
        x1, y1 = batch1
        x2, y2 = batch2
        
        assert x1.shape == (2, 10)
        assert x2.shape == (2, 10)
        assert not torch.equal(x1, x2)  # Different batches


class TestBatchFunction:
    """Test the get_batch utility function."""
    
    def test_get_batch_device_transfer(self, tmp_path):
        """Test that get_batch moves tensors to correct device."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(200, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataloader, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=True, num_workers=0
        )
        
        infinite_loader = InfiniteDataLoader(dataloader)
        
        # Test CPU device
        x, y = get_batch(infinite_loader, "cpu")
        assert x.device.type == "cpu"
        assert y.device.type == "cpu"
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            x, y = get_batch(infinite_loader, "cuda")
            assert x.device.type == "cuda"
            assert y.device.type == "cuda"
    
    def test_get_batch_shape_preservation(self, tmp_path):
        """Test that get_batch preserves tensor shapes."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(500, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataloader, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=8, seq_len=64, train=True, num_workers=0
        )
        
        infinite_loader = InfiniteDataLoader(dataloader)
        x, y = get_batch(infinite_loader, "cpu")
        
        assert x.shape == (8, 64)
        assert y.shape == (8, 64)


class TestMultiGPUCompatibility:
    """Test dataloader compatibility with multi-GPU setups."""
    
    def test_dataloader_with_dataparallel(self, tmp_path):
        """Test that dataloader works with DataParallel models."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        # Setup test data
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        from model import create_model
        
        # Create model and dataloader
        dataloader, vocab_size = create_dataloader(
            str(tmp_path / "data"), batch_size=8, seq_len=32, train=True, num_workers=0
        )
        
        model = create_model(vocab_size).cuda()
        model_parallel = nn.DataParallel(model)
        
        # Test that batches work with DataParallel
        x, y = next(iter(dataloader))
        x, y = x.cuda(), y.cuda()
        
        model_parallel.eval()
        with torch.no_grad():
            logits = model_parallel(x)
        
        assert logits.shape == (8, 32, vocab_size)
        assert logits.device.type == "cuda"
    
    def test_multi_gpu_batch_splitting(self, tmp_path):
        """Test that DataParallel properly splits batches across GPUs."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(2000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        from model import create_model
        
        # Large batch size that will be split across GPUs
        batch_size = 16
        dataloader, vocab_size = create_dataloader(
            str(tmp_path / "data"), batch_size=batch_size, seq_len=32, train=True, num_workers=0
        )
        
        model = create_model(vocab_size).cuda()
        model_parallel = nn.DataParallel(model)
        
        infinite_loader = InfiniteDataLoader(dataloader)
        x, y = get_batch(infinite_loader, "cuda")
        
        # Forward pass should work with batch splitting
        model_parallel.train()
        logits, loss = model_parallel(x, y)
        
        # Handle DataParallel loss gathering
        if loss.dim() > 0:
            loss = loss.mean()
        
        loss.backward()
        
        assert logits.shape == (batch_size, 32, vocab_size)
        assert loss.item() > 0
    
    def test_dataloader_memory_efficiency(self, tmp_path):
        """Test dataloader memory usage with large batches."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA tests require GPU")
        
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(5000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        # Test with pin_memory for efficiency
        dataloader, _ = create_dataloader(
            str(tmp_path / "data"), 
            batch_size=32, 
            seq_len=128, 
            train=True, 
            num_workers=0
        )
        
        assert dataloader.pin_memory == True
        
        # Check that batches transfer efficiently
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()
        
        x, y = next(iter(dataloader))
        x, y = x.cuda(), y.cuda()
        
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        
        # Memory should have increased
        assert end_memory > start_memory
        
        # Clean up
        del x, y
        torch.cuda.empty_cache()
    
    def test_dataloader_with_different_batch_sizes(self, tmp_path):
        """Test dataloader with various batch sizes for multi-GPU scenarios."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(3000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        # Test different batch sizes that are common in multi-GPU setups
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            dataloader, vocab_size = create_dataloader(
                str(tmp_path / "data"), 
                batch_size=batch_size, 
                seq_len=64, 
                train=True, 
                num_workers=0
            )
            
            x, y = next(iter(dataloader))
            
            assert x.shape == (batch_size, 64)
            assert y.shape == (batch_size, 64)
            assert vocab_size > 0


class TestDataLoaderIntegration:
    """Integration tests for dataloader with training scenarios."""
    
    def test_training_loop_simulation(self, tmp_path):
        """Test dataloader in a simulated training loop."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(1000, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_train.bin")
        
        dataloader, vocab_size = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=True, num_workers=0
        )
        
        from model import create_model
        
        model = create_model(vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        infinite_loader = InfiniteDataLoader(dataloader)
        
        # Simulate training steps
        model.train()
        losses = []
        
        for step in range(10):
            x, y = get_batch(infinite_loader, "cpu")
            
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            losses.append(loss.item())
        
        # Loss should be reasonable
        assert all(loss > 0 for loss in losses)
        assert len(losses) == 10
    
    def test_dataloader_deterministic_behavior(self, tmp_path):
        """Test deterministic behavior for reproducible training."""
        os.makedirs(tmp_path / "data", exist_ok=True)
        tokens = np.arange(500, dtype=np.uint16)
        tokens.tofile(tmp_path / "data" / "tinystories_val.bin")  # Use val for deterministic
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        dataloader1, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=False, num_workers=0
        )
        
        torch.manual_seed(42)
        
        dataloader2, _ = create_dataloader(
            str(tmp_path / "data"), batch_size=4, seq_len=32, train=False, num_workers=0
        )
        
        # Should produce identical batches
        x1, y1 = next(iter(dataloader1))
        x2, y2 = next(iter(dataloader2))
        
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 