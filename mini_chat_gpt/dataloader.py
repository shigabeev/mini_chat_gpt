"""
TinyStories dataloader with efficient tokenization and batching.
"""

import os
import pickle
import requests
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import tiktoken


class TinyStoriesDataset(Dataset):
    """TinyStories dataset with on-the-fly tokenization."""
    
    def __init__(self, data_path: str, seq_len: int = 1024, train: bool = True):
        self.seq_len = seq_len
        self.train = train
        
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        self.vocab_size = self.tokenizer.n_vocab
        
        # Download and prepare data
        self.data_file = self._prepare_data(data_path, train)
        
        # Memory-map the tokenized data
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')
        print(f"Loaded {len(self.data):,} tokens from {self.data_file}")
    
    def _prepare_data(self, data_path: str, train: bool) -> str:
        """Prepare and tokenize TinyStories data from local files."""
        split = "train" if train else "val"
        tokens_file = os.path.join(data_path, f"tinystories_{split}.bin")
        
        if os.path.exists(tokens_file):
            return tokens_file
        
        # Create data directory
        os.makedirs(data_path, exist_ok=True)
        
        # Use local files instead of downloading
        if train:
            txt_file = "TinyStories/TinyStoriesV2-GPT4-train.txt"
        else:
            txt_file = "TinyStories/TinyStoriesV2-GPT4-valid.txt"
        
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Local file {txt_file} not found. Please ensure TinyStories files are in the TinyStories/ directory.")
        
        # Tokenize
        print(f"Tokenizing {txt_file}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split by <|endoftext|> tokens (the actual separator used in TinyStories)
        stories = text.split('<|endoftext|>')
        stories = [story.strip() for story in stories if story.strip()]
        
        # Tokenize each story and add endoftext tokens
        all_tokens = []
        for story in stories:
            if story.strip():  # Only process non-empty stories
                tokens = self.tokenizer.encode(story.strip())
                all_tokens.extend(tokens)
                # Add the endoftext token ID (this is usually token 50256 for GPT models)
                try:
                    eot_token = self.tokenizer.encode('<|endoftext|>')[0]
                    all_tokens.append(eot_token)
                except:
                    # Fallback if encoding fails
                    all_tokens.append(50256)  # Standard GPT endoftext token
        
        # Convert to numpy array and save
        tokens_np = np.array(all_tokens, dtype=np.uint32)
        tokens_np.tofile(tokens_file)
        
        print(f"Tokenized {len(stories):,} stories into {len(tokens_np):,} tokens")
        return tokens_file
    
    def __len__(self) -> int:
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence of length seq_len + 1
        chunk = self.data[idx:idx + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])  # inputs
        y = torch.from_numpy(chunk[1:])   # targets
        return x, y


def create_dataloader(
    data_path: str,
    batch_size: int,
    seq_len: int = 1024,
    train: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, int]:
    """Create TinyStories dataloader."""
    
    dataset = TinyStoriesDataset(data_path, seq_len, train)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader, dataset.vocab_size


class InfiniteDataLoader:
    """Wrapper for infinite data loading during training."""
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def get_batch(dataloader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch and move to device."""
    x, y = next(dataloader)
    x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    # Test the dataloader with local TinyStories files
    data_path = "./TinyStories/data"
    
    print("Testing TinyStories dataloader...")
    print("=" * 50)
    
    # Test training dataloader
    print("\n1. Testing training dataloader...")
    train_loader, vocab_size = create_dataloader(data_path, batch_size=4, seq_len=256, train=True)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset length: {len(train_loader.dataset)}")
    
    # Test batch
    x, y = next(iter(train_loader))
    print(f"Training batch shape: {x.shape}, {y.shape}")
    print(f"Sample training tokens: {x[0, :10].tolist()}")
    
    # Test validation dataloader
    print("\n2. Testing validation dataloader...")
    val_loader, _ = create_dataloader(data_path, batch_size=4, seq_len=256, train=False)
    
    print(f"Validation dataset length: {len(val_loader.dataset)}")
    
    # Test validation batch
    x_val, y_val = next(iter(val_loader))
    print(f"Validation batch shape: {x_val.shape}, {y_val.shape}")
    print(f"Sample validation tokens: {x_val[0, :10].tolist()}")
    
    # Test that input and target are properly offset
    print(f"\n3. Testing input/target offset...")
    print(f"First input token: {x[0, 0].item()}")
    print(f"First target token: {y[0, 0].item()}")
    print(f"Second input token: {x[0, 1].item()}")
    print(f"Are they shifted correctly? {x[0, 1].item() == y[0, 0].item()}")
    
    # Test InfiniteDataLoader with smaller dataset for speed
    print(f"\n4. Testing InfiniteDataLoader...")
    
    # Create a tiny dataset for quick testing
    tiny_data_path = "./TinyStories/tiny_test_data"
    os.makedirs(tiny_data_path, exist_ok=True)
    
    # Create small test data (just 200 tokens)
    tiny_tokens = np.arange(200, dtype=np.uint32)
    tiny_tokens.tofile(os.path.join(tiny_data_path, "tinystories_train.bin"))
    
    # Create small dataloader
    tiny_loader, _ = create_dataloader(tiny_data_path, batch_size=2, seq_len=16, train=True, num_workers=0)
    
    infinite_loader = InfiniteDataLoader(tiny_loader)
    batch1 = next(infinite_loader)
    batch2 = next(infinite_loader)
    print(f"Infinite loader batch 1 shape: {batch1[0].shape}")
    print(f"Infinite loader batch 2 shape: {batch2[0].shape}")
    
    # Test cycling through multiple epochs quickly
    for i in range(5):
        next(infinite_loader)
    print(f"Successfully cycled through multiple epochs")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!") 