"""
TinyStories dataloader with efficient tokenization and batching.
"""

import os
import pickle
import requests
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import tiktoken


class TinyStoriesDataset(Dataset):
    """TinyStories dataset with on-the-fly tokenization."""
    
    def __init__(self, data_path: str, seq_len: int = 1024, train: bool = True):
        self.seq_len = seq_len
        self.train = train
        
        # Use GPT-2 tokenizer (cl100k_base for better vocab efficiency)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            # Fallback to gpt2
            self.tokenizer = tiktoken.get_encoding("gpt2")
        
        self.vocab_size = self.tokenizer.n_vocab
        
        # Download and prepare data
        self.data_file = self._prepare_data(data_path, train)
        
        # Memory-map the tokenized data
        self.data = np.memmap(self.data_file, dtype=np.uint16, mode='r')
        print(f"Loaded {len(self.data):,} tokens from {self.data_file}")
    
    def _prepare_data(self, data_path: str, train: bool) -> str:
        """Download and tokenize TinyStories data if needed."""
        split = "train" if train else "val"
        tokens_file = os.path.join(data_path, f"tinystories_{split}.bin")
        
        if os.path.exists(tokens_file):
            return tokens_file
        
        # Create data directory
        os.makedirs(data_path, exist_ok=True)
        
        # Download URLs
        urls = {
            "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
            "val": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        }
        
        txt_file = os.path.join(data_path, f"tinystories_{split}.txt")
        
        # Download if needed
        if not os.path.exists(txt_file):
            print(f"Downloading TinyStories {split} data...")
            response = requests.get(urls[split])
            response.raise_for_status()
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"Downloaded to {txt_file}")
        
        # Tokenize
        print(f"Tokenizing {txt_file}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Add special tokens for story boundaries
        stories = text.split('\n\n')  # Stories are separated by double newlines
        stories = [story.strip() for story in stories if story.strip()]
        
        # Tokenize each story and add EOS tokens
        all_tokens = []
        for story in stories:
            tokens = self.tokenizer.encode(story)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eot_token)  # End of story token
        
        # Convert to numpy array and save
        tokens_np = np.array(all_tokens, dtype=np.uint16)
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
    # Test the dataloader
    data_path = "./TinyStories/data"
    train_loader, vocab_size = create_dataloader(data_path, batch_size=4, seq_len=256, train=True)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Dataset length: {len(train_loader.dataset)}")
    
    # Test batch
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, {y.shape}")
    print(f"Sample tokens: {x[0, :10].tolist()}") 