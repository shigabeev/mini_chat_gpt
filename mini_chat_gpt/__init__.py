"""
Mini ChatGPT: A clean, minimal implementation of GPT-2 for pretraining on TinyStories.

This package provides a modern implementation of GPT-2 with enhancements like:
- FlashAttention for memory-efficient attention computation
- RoPE (Rotary Position Embedding) for better position encoding
- Pre-Norm LayerNorm for improved training stability
- SwiGLU activation with parallel FFN computation
- Mixed Precision Training for faster training
"""

__version__ = "0.1.0"
__author__ = "Mini ChatGPT Team"
__email__ = ""

# Import main classes and functions for easy access
from .model import GPT, create_model, Attention, Block, SwiGLU
from .dataloader import TinyStoriesDataset, create_dataloader, InfiniteDataLoader, get_batch

__all__ = [
    # Model components
    "GPT",
    "create_model", 
    "Attention",
    "Block",
    "SwiGLU",
    # Data components
    "TinyStoriesDataset",
    "create_dataloader",
    "InfiniteDataLoader", 
    "get_batch",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
] 