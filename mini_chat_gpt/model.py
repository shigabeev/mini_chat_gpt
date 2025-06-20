"""
GPT-2 style model with modern enhancements.
Features: FlashAttention, Pre-Norm LayerNorm, SwiGLU+parallel FFN, RoPE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("FlashAttention not available, falling back to standard attention")


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # xq and xk have shape: (B, T, n_heads, head_dim)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # xq_ and xk_ now have shape: (B, T, n_heads, head_dim//2)
    
    seq_len = xq_.shape[1]
    expected_dim = xq_.shape[-1]  # head_dim//2
    
    # Handle freqs_cis tensor that might have been converted from complex to real by DataParallel
    if freqs_cis.dtype == torch.complex64:
        # Normal complex tensor case
        if freqs_cis.dim() == 2 and freqs_cis.shape[0] >= seq_len and freqs_cis.shape[1] == expected_dim:
            freqs_cis = freqs_cis[:seq_len]  # Shape: (T, head_dim//2)
        else:
            raise RuntimeError(f"Complex freqs_cis shape mismatch: got {freqs_cis.shape}, need at least ({seq_len}, {expected_dim})")
    elif freqs_cis.dtype.is_floating_point and freqs_cis.dim() == 3 and freqs_cis.shape[-1] == 2:
        # DataParallel converted complex to real with shape (..., 2) for (real, imag)
        if freqs_cis.shape[0] >= seq_len and freqs_cis.shape[1] == expected_dim:
            # Shape is (seq_len, head_dim//2, 2) - convert back to complex
            freqs_cis = freqs_cis[:seq_len]  # Take sequence slice first
            freqs_cis = torch.view_as_complex(freqs_cis)  # Convert back to complex
        else:
            raise RuntimeError(f"Real freqs_cis shape mismatch: got {freqs_cis.shape}, need at least ({seq_len}, {expected_dim}, 2)")
    else:
        # Handle other edge cases
        if freqs_cis.numel() >= seq_len * expected_dim * 2:  # Account for complex = 2x real elements
            # Try to reshape to complex format
            total_elements = freqs_cis.numel()
            if total_elements == seq_len * expected_dim * 2:
                freqs_cis = freqs_cis.view(seq_len, expected_dim, 2)
                freqs_cis = torch.view_as_complex(freqs_cis)
            else:
                # Take first part and reshape
                needed_elements = seq_len * expected_dim * 2
                freqs_cis = freqs_cis.flatten()[:needed_elements].view(seq_len, expected_dim, 2)
                freqs_cis = torch.view_as_complex(freqs_cis)
        else:
            raise RuntimeError(f"freqs_cis insufficient elements: got {freqs_cis.numel()}, need {seq_len * expected_dim * 2} for complex")
    
    # Ensure we have the right complex tensor shape
    if not freqs_cis.dtype == torch.complex64 or freqs_cis.shape != (seq_len, expected_dim):
        raise RuntimeError(f"freqs_cis final error: got shape {freqs_cis.shape} dtype {freqs_cis.dtype}, expected ({seq_len}, {expected_dim}) complex64")
    
    # Reshape freqs_cis to broadcast properly: (1, T, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    """SwiGLU activation with parallel computation."""
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with FlashAttention and RoPE."""
    
    def __init__(self, dim: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
        # RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_seq_len))
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Linear projections
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE - pass full freqs_cis and let apply_rotary_emb handle slicing
        q, k = apply_rotary_emb(q, k, self.freqs_cis)
        
        if FLASH_ATTN_AVAILABLE and q.is_cuda:
            # FlashAttention expects (B, T, H, D) and only works on CUDA with fp16/bf16
            input_dtype = q.dtype
            if input_dtype not in [torch.float16, torch.bfloat16]:
                # Convert to bfloat16 for FlashAttention
                q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
            
            out = flash_attn_func(q, k, v, causal=True)
            out = out.contiguous().view(B, T, C)
            
            # Convert back to original dtype if needed
            if input_dtype not in [torch.float16, torch.bfloat16]:
                out = out.to(input_dtype)
        else:
            # Standard attention
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D)
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        
        return self.wo(out)


class Block(nn.Module):
    """Transformer block with Pre-Norm LayerNorm."""
    
    def __init__(self, dim: int, n_heads: int, max_seq_len: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT model with modern enhancements."""
    
    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 1024,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, max_seq_len, mlp_ratio) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 paper."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        # Token embeddings
        x = self.tok_emb(idx)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and head
        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is not None:
            # Compute loss with proper reduction for DataParallel
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1, 
                reduction='mean'
            )
            # Ensure loss is always a scalar by taking mean if it's not
            if loss.dim() > 0:
                loss = loss.mean()
            return logits, loss
        else:
            return logits, None
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, stop_tokens: Optional[list] = None) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop sequence if needed
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for stop tokens
            if stop_tokens is not None:
                # Get the last generated token for each sequence in the batch
                last_tokens = idx_next.squeeze(-1)  # Shape: (batch_size,)
                
                # Check if any of the last tokens are stop tokens
                for stop_token in stop_tokens:
                    if torch.any(last_tokens == stop_token):
                        # Stop generation if any sequence hits a stop token
                        return idx
        
        return idx
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(vocab_size: int) -> GPT:
    """Create a ~100M parameter GPT model."""
    return GPT(
        vocab_size=vocab_size,
        dim=768,
        n_layers=12,
        n_heads=12,
        max_seq_len=1024,
        mlp_ratio=4.0,
    ) 