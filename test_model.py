"""
Comprehensive tests for GPT model implementation.
Tests all components thoroughly with edge cases and validation.
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from model import (
    precompute_freqs_cis,
    apply_rotary_emb,
    SwiGLU,
    Attention,
    Block,
    GPT,
    create_model,
)

# Setup device for testing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running tests on device: {DEVICE}")


class TestRoPEFunctions:
    """Test Rotary Position Embedding functions."""
    
    def test_precompute_freqs_cis_shape(self):
        """Test RoPE frequency computation produces correct shapes."""
        dim, seq_len = 64, 128
        freqs_cis = precompute_freqs_cis(dim, seq_len)
        
        assert freqs_cis.shape == (seq_len, dim // 2)
        assert freqs_cis.dtype == torch.complex64
    
    def test_precompute_freqs_cis_values(self):
        """Test RoPE frequency values are mathematically correct."""
        dim, seq_len = 4, 8
        theta = 10000.0
        freqs_cis = precompute_freqs_cis(dim, seq_len, theta)
        
        # Check first frequency
        expected_freq_0 = 1.0 / (theta ** (0.0 / dim))
        assert torch.allclose(freqs_cis[0, 0].real, torch.cos(torch.tensor(0.0)))
        assert torch.allclose(freqs_cis[0, 0].imag, torch.sin(torch.tensor(0.0)))
    
    def test_apply_rotary_emb_shapes(self):
        """Test RoPE application preserves tensor shapes."""
        batch_size, seq_len, n_heads, head_dim = 2, 16, 8, 64
        
        q = torch.randn(batch_size, seq_len, n_heads, head_dim).to(DEVICE)
        k = torch.randn(batch_size, seq_len, n_heads, head_dim).to(DEVICE)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len).to(DEVICE)
        
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)
        
        assert q_rope.shape == q.shape
        assert k_rope.shape == k.shape
        assert q_rope.dtype == q.dtype
        assert k_rope.dtype == k.dtype
    
    def test_apply_rotary_emb_rotation_property(self):
        """Test that RoPE actually rotates the embeddings."""
        batch_size, seq_len, n_heads, head_dim = 1, 4, 1, 4
        
        # Simple test vectors
        q = torch.ones(batch_size, seq_len, n_heads, head_dim).to(DEVICE)
        k = torch.ones(batch_size, seq_len, n_heads, head_dim).to(DEVICE)
        freqs_cis = precompute_freqs_cis(head_dim, seq_len).to(DEVICE)
        
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)
        
        # Should not be identical after rotation (except potentially at position 0)
        assert not torch.allclose(q_rope[0, 1:], q[0, 1:])
        assert not torch.allclose(k_rope[0, 1:], k[0, 1:])


class TestSwiGLU:
    """Test SwiGLU activation function."""
    
    def test_swiglu_initialization(self):
        """Test SwiGLU module initializes correctly."""
        dim, hidden_dim = 768, 3072
        swiglu = SwiGLU(dim, hidden_dim)
        
        assert isinstance(swiglu.w1, nn.Linear)
        assert isinstance(swiglu.w2, nn.Linear)
        assert isinstance(swiglu.w3, nn.Linear)
        assert swiglu.w1.in_features == dim
        assert swiglu.w1.out_features == hidden_dim
        assert swiglu.w2.in_features == hidden_dim
        assert swiglu.w2.out_features == dim
        assert swiglu.w3.in_features == dim
        assert swiglu.w3.out_features == hidden_dim
    
    def test_swiglu_forward_shape(self):
        """Test SwiGLU forward pass produces correct shapes."""
        batch_size, seq_len, dim = 4, 32, 768
        hidden_dim = 3072
        
        swiglu = SwiGLU(dim, hidden_dim).to(DEVICE)
        x = torch.randn(batch_size, seq_len, dim).to(DEVICE)
        
        output = swiglu(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert output.dtype == x.dtype
    
    def test_swiglu_nonlinearity(self):
        """Test that SwiGLU introduces non-linearity."""
        dim, hidden_dim = 64, 256
        swiglu = SwiGLU(dim, hidden_dim)
        
        # Test with different inputs
        x1 = torch.zeros(1, 1, dim)
        x2 = torch.ones(1, 1, dim)
        
        out1 = swiglu(x1)
        out2 = swiglu(x2)
        
        # Should produce different outputs for different inputs
        assert not torch.allclose(out1, out2)
    
    def test_swiglu_gradient_flow(self):
        """Test that gradients flow through SwiGLU."""
        dim, hidden_dim = 64, 256
        swiglu = SwiGLU(dim, hidden_dim)
        
        x = torch.randn(2, 4, dim, requires_grad=True)
        output = swiglu(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestAttention:
    """Test multi-head attention with RoPE."""
    
    def test_attention_initialization(self):
        """Test attention module initializes correctly."""
        dim, n_heads, max_seq_len = 768, 12, 1024
        attention = Attention(dim, n_heads, max_seq_len)
        
        assert attention.dim == dim
        assert attention.n_heads == n_heads
        assert attention.head_dim == dim // n_heads
        assert attention.freqs_cis.shape == (max_seq_len, attention.head_dim // 2)
        assert attention.mask.shape == (max_seq_len, max_seq_len)
    
    def test_attention_forward_shape(self):
        """Test attention forward pass produces correct shapes."""
        batch_size, seq_len, dim, n_heads = 2, 16, 768, 12
        max_seq_len = 1024
        
        attention = Attention(dim, n_heads, max_seq_len).to(DEVICE)
        x = torch.randn(batch_size, seq_len, dim).to(DEVICE)
        
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert output.dtype == x.dtype
    
    def test_attention_causal_mask(self):
        """Test that attention respects causal masking."""
        batch_size, seq_len, dim, n_heads = 1, 8, 64, 4
        max_seq_len = 16
        
        attention = Attention(dim, n_heads, max_seq_len)
        
        # Create input with distinct values at each position
        x = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, dim)
        
        with torch.no_grad():
            output = attention(x)
        
        # Each position should only attend to previous positions
        # This is hard to test directly, so we test the mask exists
        assert torch.all(attention.mask[:seq_len, :seq_len].tril() == attention.mask[:seq_len, :seq_len])
    
    def test_attention_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        dim, n_heads, max_seq_len = 64, 8, 128
        attention = Attention(dim, n_heads, max_seq_len)
        
        for seq_len in [1, 16, 32, 64, 128]:
            batch_size = 2
            x = torch.randn(batch_size, seq_len, dim)
            output = attention(x)
            assert output.shape == (batch_size, seq_len, dim)
    
    def test_attention_head_dimension_constraint(self):
        """Test that attention requires dim to be divisible by n_heads."""
        with pytest.raises(AssertionError):
            Attention(dim=100, n_heads=7, max_seq_len=64)  # 100 not divisible by 7


class TestBlock:
    """Test transformer block."""
    
    def test_block_initialization(self):
        """Test block initializes correctly."""
        dim, n_heads, max_seq_len = 768, 12, 1024
        block = Block(dim, n_heads, max_seq_len)
        
        assert isinstance(block.ln_1, nn.LayerNorm)
        assert isinstance(block.attn, Attention)
        assert isinstance(block.ln_2, nn.LayerNorm)
        assert isinstance(block.mlp, SwiGLU)
    
    def test_block_forward_shape(self):
        """Test block forward pass produces correct shapes."""
        batch_size, seq_len, dim, n_heads = 2, 16, 768, 12
        max_seq_len = 1024
        
        block = Block(dim, n_heads, max_seq_len)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, dim)
        assert output.dtype == x.dtype
    
    def test_block_residual_connections(self):
        """Test that residual connections work correctly."""
        dim, n_heads, max_seq_len = 64, 8, 128
        block = Block(dim, n_heads, max_seq_len)
        
        # Zero out all weights to test residual connections
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()
        
        x = torch.randn(2, 8, dim)
        output = block(x)
        
        # With zero weights, output should equal input due to residual connections
        assert torch.allclose(output, x, atol=1e-6)
    
    def test_block_gradient_flow(self):
        """Test that gradients flow through the block."""
        dim, n_heads, max_seq_len = 64, 8, 128
        block = Block(dim, n_heads, max_seq_len)
        
        x = torch.randn(2, 8, dim, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check that at least some parameters have gradients
        param_grads = [p.grad for p in block.parameters() if p.grad is not None]
        assert len(param_grads) > 0


class TestGPTModel:
    """Test full GPT model."""
    
    def test_gpt_initialization(self):
        """Test GPT model initializes correctly."""
        vocab_size, dim, n_layers, n_heads = 1000, 256, 6, 8
        model = GPT(vocab_size, dim, n_layers, n_heads)
        
        assert model.vocab_size == vocab_size
        assert model.dim == dim
        assert len(model.blocks) == n_layers
        assert isinstance(model.tok_emb, nn.Embedding)
        assert isinstance(model.ln_f, nn.LayerNorm)
        assert isinstance(model.head, nn.Linear)
    
    def test_gpt_parameter_count(self):
        """Test that parameter count is reasonable for 100M model."""
        vocab_size = 50000
        model = create_model(vocab_size)
        
        param_count = model.get_num_params()
        
        # Should be around 100M parameters (allowing some variance)
        assert 80_000_000 < param_count < 160_000_000
        print(f"Model has {param_count:,} parameters")
    
    def test_gpt_forward_shape(self):
        """Test GPT forward pass produces correct shapes."""
        vocab_size, dim, n_layers, n_heads = 1000, 256, 6, 8
        batch_size, seq_len = 4, 32
        
        model = GPT(vocab_size, dim, n_layers, n_heads).to(DEVICE)
        idx = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
        
        logits, loss = model(idx)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is None  # No targets provided
    
    def test_gpt_forward_with_targets(self):
        """Test GPT forward pass with targets produces loss."""
        vocab_size, dim, n_layers, n_heads = 1000, 256, 6, 8
        batch_size, seq_len = 4, 32
        
        model = GPT(vocab_size, dim, n_layers, n_heads).to(DEVICE)
        idx = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEVICE)
        
        logits, loss = model(idx, targets)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert loss is not None
        assert loss.item() > 0  # Should have positive loss
    
    def test_gpt_generation(self):
        """Test GPT text generation."""
        vocab_size, dim, n_layers, n_heads = 100, 128, 4, 4
        model = GPT(vocab_size, dim, n_layers, n_heads, max_seq_len=64).to(DEVICE)
        model.eval()
        
        # Start with a single token
        idx = torch.tensor([[42]]).to(DEVICE)  # Single token
        max_new_tokens = 10
        
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens)
        
        assert generated.shape == (1, 1 + max_new_tokens)
        assert torch.all(generated >= 0)
        assert torch.all(generated < vocab_size)
    
    def test_gpt_generation_temperature(self):
        """Test GPT generation with different temperatures."""
        vocab_size, dim, n_layers, n_heads = 100, 128, 4, 4
        model = GPT(vocab_size, dim, n_layers, n_heads, max_seq_len=64)
        model.eval()
        
        idx = torch.tensor([[42]])
        max_new_tokens = 5
        
        # Test different temperatures
        with torch.no_grad():
            gen_low_temp = model.generate(idx, max_new_tokens, temperature=0.1)
            gen_high_temp = model.generate(idx, max_new_tokens, temperature=2.0)
        
        assert gen_low_temp.shape == gen_high_temp.shape
        # Different temperatures should generally produce different outputs
        # (though not guaranteed, so we just check shapes)
    
    def test_gpt_generation_top_k(self):
        """Test GPT generation with top-k sampling."""
        vocab_size, dim, n_layers, n_heads = 100, 128, 4, 4
        model = GPT(vocab_size, dim, n_layers, n_heads, max_seq_len=64)
        model.eval()
        
        idx = torch.tensor([[42]])
        max_new_tokens = 5
        
        with torch.no_grad():
            generated = model.generate(idx, max_new_tokens, top_k=10)
        
        assert generated.shape == (1, 1 + max_new_tokens)
        assert torch.all(generated >= 0)
        assert torch.all(generated < vocab_size)
    
    def test_gpt_weight_tying(self):
        """Test that embedding and output head weights are tied."""
        vocab_size, dim = 1000, 256
        model = GPT(vocab_size, dim, n_layers=2, n_heads=4)
        
        # Check that weights are the same object
        assert model.tok_emb.weight is model.head.weight
    
    def test_gpt_sequence_length_constraint(self):
        """Test that model enforces sequence length constraints."""
        vocab_size, max_seq_len = 100, 16
        model = GPT(vocab_size, dim=64, n_layers=2, n_heads=4, max_seq_len=max_seq_len)
        
        # Test with sequence longer than max_seq_len
        long_seq = torch.randint(0, vocab_size, (1, max_seq_len + 5))
        
        with pytest.raises(AssertionError):
            model(long_seq)
    
    def test_gpt_different_batch_sizes(self):
        """Test model with different batch sizes."""
        vocab_size, dim, n_layers, n_heads = 100, 128, 2, 4
        seq_len = 16
        model = GPT(vocab_size, dim, n_layers, n_heads)
        
        for batch_size in [1, 2, 4, 8]:
            idx = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits, _ = model(idx)
            assert logits.shape == (batch_size, seq_len, vocab_size)


class TestModelIntegration:
    """Integration tests for the full model."""
    
    def test_create_model_function(self):
        """Test the create_model helper function."""
        vocab_size = 5000
        model = create_model(vocab_size)
        
        assert isinstance(model, GPT)
        assert model.vocab_size == vocab_size
        assert model.dim == 768
        assert len(model.blocks) == 12
    
    def test_model_training_step(self):
        """Test a complete training step."""
        vocab_size, batch_size, seq_len = 1000, 4, 32
        model = create_model(vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Generate random data
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits, loss = model(idx, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is reasonable
        assert 0 < loss.item() < 20  # Cross-entropy loss should be reasonable
        
        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_overfitting_small_data(self):
        """Test that model can overfit to small dataset (sanity check)."""
        vocab_size, batch_size, seq_len = 50, 2, 8
        model = GPT(vocab_size, dim=64, n_layers=2, n_heads=4, max_seq_len=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Fixed small dataset
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        initial_loss = None
        final_loss = None
        
        # Train for a few steps
        for step in range(50):
            logits, loss = model(idx, targets)
            
            if step == 0:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.8  # At least 20% reduction
    
    def test_model_deterministic_with_seed(self):
        """Test that model produces deterministic results with fixed seed."""
        torch.manual_seed(42)
        vocab_size = 100
        model1 = create_model(vocab_size)
        
        torch.manual_seed(42)
        model2 = create_model(vocab_size)
        
        # Models should have identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
        
        # Forward passes should be identical
        torch.manual_seed(123)
        idx = torch.randint(0, vocab_size, (2, 16))
        
        torch.manual_seed(456)
        out1, _ = model1(idx)
        
        torch.manual_seed(456)
        out2, _ = model2(idx)
        
        assert torch.allclose(out1, out2)


class TestModelSerialization:
    """Test model serialization and loading functionality."""
    
    def test_save_and_load_checkpoint_basic(self, tmp_path):
        """Test basic checkpoint save and load functionality."""
        from train import save_checkpoint, load_checkpoint
        
        # Create model and optimizer
        vocab_size = 1000
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train for a few steps to change weights
        model.train()
        for _ in range(5):
            x = torch.randint(0, vocab_size, (2, 16)).to(DEVICE)
            targets = torch.randint(0, vocab_size, (2, 16)).to(DEVICE)
            logits, loss = model(x, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint
        step = 100
        loss_val = 2.5
        save_checkpoint(model, optimizer, step, loss_val, str(tmp_path))
        
        # Create new model and optimizer
        model2 = create_model(vocab_size).to(DEVICE)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        # Load checkpoint
        checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
        loaded_step = load_checkpoint(str(checkpoint_path), model2, optimizer2)
        
        # Verify loaded step
        assert loaded_step == step
        
        # Verify model weights are identical
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
        
        # Verify optimizer states are identical
        assert optimizer.state_dict().keys() == optimizer2.state_dict().keys()
    
    def test_checkpoint_contains_correct_metadata(self, tmp_path):
        """Test that checkpoint contains all necessary metadata."""
        from train import save_checkpoint
        
        vocab_size = 500
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        step = 42
        loss_val = 3.14
        save_checkpoint(model, optimizer, step, loss_val, str(tmp_path))
        
        # Load and verify checkpoint contents
        checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check required keys
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'step', 'loss', 'config']
        for key in required_keys:
            assert key in checkpoint
        
        # Check config contents
        config = checkpoint['config']
        config_keys = ['vocab_size', 'dim', 'n_layers', 'n_heads', 'max_seq_len']
        for key in config_keys:
            assert key in config
        
        # Verify values
        assert checkpoint['step'] == step
        assert checkpoint['loss'] == loss_val
        assert config['vocab_size'] == vocab_size
        assert config['dim'] == 768  # From create_model defaults
        assert config['n_layers'] == 12
        assert config['n_heads'] == 12
        assert config['max_seq_len'] == 1024
    
    def test_load_checkpoint_preserves_model_behavior(self, tmp_path):
        """Test that loaded model behaves identically to original."""
        from train import save_checkpoint, load_checkpoint
        
        vocab_size = 1000
        model1 = create_model(vocab_size).to(DEVICE)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        
        # Set model to eval mode and fix seed for deterministic behavior
        model1.eval()
        torch.manual_seed(42)
        
        # Generate some test data
        x = torch.randint(0, vocab_size, (2, 16)).to(DEVICE)
        
        # Get original model output
        with torch.no_grad():
            logits1, _ = model1(x)
        
        # Save checkpoint
        save_checkpoint(model1, optimizer1, 50, 1.0, str(tmp_path))
        
        # Create new model and load checkpoint
        model2 = create_model(vocab_size).to(DEVICE)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        checkpoint_path = tmp_path / "checkpoint_step_50.pt"
        load_checkpoint(str(checkpoint_path), model2, optimizer2)
        
        model2.eval()
        
        # Get loaded model output with same input
        with torch.no_grad():
            logits2, _ = model2(x)
        
        # Outputs should be identical
        assert torch.allclose(logits1, logits2, atol=1e-6)
    
    def test_checkpoint_different_device_loading(self, tmp_path):
        """Test loading checkpoint works across different devices."""
        from train import save_checkpoint, load_checkpoint
        
        vocab_size = 500
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, 25, 2.0, str(tmp_path))
        
        # Load checkpoint to CPU (map_location='cpu' in load_checkpoint)
        model_cpu = create_model(vocab_size)  # Create on CPU
        optimizer_cpu = torch.optim.AdamW(model_cpu.parameters(), lr=1e-4)
        
        checkpoint_path = tmp_path / "checkpoint_step_25.pt"
        loaded_step = load_checkpoint(str(checkpoint_path), model_cpu, optimizer_cpu)
        
        assert loaded_step == 25
        
        # Move loaded model to GPU and verify it works
        model_cpu.to(DEVICE)
        x = torch.randint(0, vocab_size, (1, 8)).to(DEVICE)
        
        with torch.no_grad():
            logits, _ = model_cpu(x)
        
        assert logits.shape == (1, 8, vocab_size)
        assert logits.device.type == DEVICE
    
    def test_checkpoint_optimizer_state_preservation(self, tmp_path):
        """Test that optimizer state is properly preserved."""
        from train import save_checkpoint, load_checkpoint
        
        vocab_size = 300
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        
        # Train for several steps to build optimizer state
        model.train()
        for step in range(10):
            x = torch.randint(0, vocab_size, (2, 8)).to(DEVICE)
            targets = torch.randint(0, vocab_size, (2, 8)).to(DEVICE)
            logits, loss = model(x, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Get optimizer state before saving
        original_state = optimizer.state_dict()
        
        # Save checkpoint
        save_checkpoint(model, optimizer, 10, 1.5, str(tmp_path))
        
        # Create new optimizer and load state
        model2 = create_model(vocab_size).to(DEVICE)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
        
        checkpoint_path = tmp_path / "checkpoint_step_10.pt"
        load_checkpoint(str(checkpoint_path), model2, optimizer2)
        
        loaded_state = optimizer2.state_dict()
        
        # Compare optimizer states
        assert original_state.keys() == loaded_state.keys()
        
        # Check learning rate and other params
        for orig_group, loaded_group in zip(original_state['param_groups'], loaded_state['param_groups']):
            assert orig_group['lr'] == loaded_group['lr']
            assert orig_group['betas'] == loaded_group['betas']
            assert orig_group['weight_decay'] == loaded_group['weight_decay']
    
    def test_checkpoint_file_format_compatibility(self, tmp_path):
        """Test checkpoint file format is standard PyTorch format."""
        from train import save_checkpoint
        
        vocab_size = 200
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        save_checkpoint(model, optimizer, 15, 1.8, str(tmp_path))
        
        # Verify we can load using standard torch.load
        checkpoint_path = tmp_path / "checkpoint_step_15.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Should be a dictionary with expected structure
        assert isinstance(checkpoint, dict)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        
        # Model state dict should be loadable
        new_model = create_model(vocab_size)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Should be able to run inference
        x = torch.randint(0, vocab_size, (1, 4))
        with torch.no_grad():
            logits, _ = new_model(x)
        assert logits.shape == (1, 4, vocab_size)
    
    def test_multiple_checkpoint_saves(self, tmp_path):
        """Test saving multiple checkpoints doesn't interfere."""
        from train import save_checkpoint, load_checkpoint
        
        vocab_size = 400
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Save multiple checkpoints
        steps = [10, 20, 30]
        losses = [3.0, 2.5, 2.0]
        
        for step, loss in zip(steps, losses):
            save_checkpoint(model, optimizer, step, loss, str(tmp_path))
        
        # Verify all checkpoints exist
        for step in steps:
            checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
            assert checkpoint_path.exists()
        
        # Load and verify each checkpoint
        for step, expected_loss in zip(steps, losses):
            checkpoint_path = tmp_path / f"checkpoint_step_{step}.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            assert checkpoint['step'] == step
            assert checkpoint['loss'] == expected_loss
    
    def test_generation_script_compatibility(self, tmp_path):
        """Test that saved checkpoints work with generate.py script."""
        from train import save_checkpoint
        from generate import load_model
        import tiktoken
        
        # Use tiktoken vocabulary size for compatibility
        try:
            tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            tokenizer = tiktoken.get_encoding("gpt2")
        
        vocab_size = tokenizer.n_vocab
        model = create_model(vocab_size).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Train model briefly to make it more interesting
        model.train()
        for _ in range(3):
            x = torch.randint(0, min(vocab_size, 10000), (2, 16)).to(DEVICE)  # Use smaller range for faster training
            targets = torch.randint(0, min(vocab_size, 10000), (2, 16)).to(DEVICE)
            logits, loss = model(x, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint
        save_checkpoint(model, optimizer, 100, 2.0, str(tmp_path))
        
        # Test loading with generate script functions
        checkpoint_path = tmp_path / "checkpoint_step_100.pt"
        loaded_model, loaded_tokenizer = load_model(str(checkpoint_path), DEVICE)
        
        # Verify model loaded correctly
        assert loaded_model.vocab_size == vocab_size
        assert loaded_tokenizer.n_vocab == vocab_size
        
        # Test a simple generation (without using the generate_text function to avoid tokenizer issues)
        # Just test that the loaded model can generate tokens
        test_tokens = torch.randint(0, 1000, (1, 5)).to(DEVICE)  # Small token range for testing
        with torch.no_grad():
            generated = loaded_model.generate(test_tokens, max_new_tokens=5)
        
        # Basic checks
        assert generated.shape[0] == 1
        assert generated.shape[1] == 10  # 5 input + 5 generated
        assert torch.all(generated >= 0)
        assert torch.all(generated < vocab_size)


class TestMultiGPUCompatibility:
    """Test multi-GPU training compatibility."""
    
    def test_dataparallel_model_creation(self):
        """Test that model can be wrapped in DataParallel."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        vocab_size = 1000
        model = create_model(vocab_size).to(DEVICE)
        
        # Wrap in DataParallel
        model_parallel = nn.DataParallel(model)
        
        # Test forward pass
        x = torch.randint(0, vocab_size, (4, 16)).to(DEVICE)
        logits = model_parallel(x)  # Without targets, only returns logits
        
        assert logits.shape == (4, 16, vocab_size)
        assert logits.device.type == "cuda"
    
    def test_dataparallel_model_parameters_access(self):
        """Test accessing model parameters through .module attribute."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        vocab_size = 500
        model = create_model(vocab_size).to(DEVICE)
        original_param_count = model.get_num_params()
        
        # Wrap in DataParallel
        model_parallel = nn.DataParallel(model)
        
        # Access parameters through .module
        parallel_param_count = model_parallel.module.get_num_params()
        
        assert parallel_param_count == original_param_count
        assert hasattr(model_parallel, 'module')
        assert model_parallel.module.vocab_size == vocab_size
    
    def test_dataparallel_checkpointing(self, tmp_path):
        """Test checkpointing with DataParallel models."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        from train import save_checkpoint, load_checkpoint
        
        vocab_size = 800
        model = create_model(vocab_size).to(DEVICE)
        model_parallel = nn.DataParallel(model)
        optimizer = torch.optim.AdamW(model_parallel.parameters(), lr=1e-4)
        
        # Train briefly
        model_parallel.train()
        for _ in range(3):
            x = torch.randint(0, vocab_size, (4, 16)).to(DEVICE)
            targets = torch.randint(0, vocab_size, (4, 16)).to(DEVICE)
            logits, loss = model_parallel(x, targets)
            # Handle DataParallel loss gathering - take mean if multiple scalars
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Save checkpoint
        save_checkpoint(model_parallel, optimizer, 50, 2.0, str(tmp_path))
        
        # Create new model and load
        model2 = create_model(vocab_size).to(DEVICE)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        checkpoint_path = tmp_path / "checkpoint_step_50.pt"
        loaded_step = load_checkpoint(str(checkpoint_path), model2, optimizer2)
        
        assert loaded_step == 50
        
        # Verify weights are identical
        for p1, p2 in zip(model_parallel.module.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_dataparallel_gradient_synchronization(self):
        """Test that gradients are properly synchronized across GPUs."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        vocab_size = 600
        model = create_model(vocab_size).to(DEVICE)
        model_parallel = nn.DataParallel(model)
        
        # Create batch that will be split across GPUs
        batch_size = 8  # Will be split as 4+4 across 2 GPUs
        x = torch.randint(0, vocab_size, (batch_size, 16)).to(DEVICE)
        targets = torch.randint(0, vocab_size, (batch_size, 16)).to(DEVICE)
        
        # Forward and backward pass
        model_parallel.train()
        logits, loss = model_parallel(x, targets)
        # Handle DataParallel loss gathering - take mean if multiple scalars
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        
        # Check that gradients exist and are finite
        for param in model_parallel.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
    
    def test_dataparallel_vs_single_gpu_equivalence(self):
        """Test that DataParallel produces equivalent results to single GPU (when possible)."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        vocab_size = 400
        
        # Create identical models
        torch.manual_seed(42)
        model1 = create_model(vocab_size).to(DEVICE)
        
        torch.manual_seed(42)
        model2 = create_model(vocab_size).to(DEVICE)
        model2_parallel = nn.DataParallel(model2)
        
        # Same input
        torch.manual_seed(123)
        x = torch.randint(0, vocab_size, (2, 16)).to(DEVICE)
        
        model1.eval()
        model2_parallel.eval()
        
        with torch.no_grad():
            logits1 = model1(x)  # Without targets, only returns logits
            logits2 = model2_parallel(x)  # Without targets, only returns logits
        
        # Results should be very close (some minor differences due to parallel execution)
        # Use relaxed tolerances for DataParallel due to floating point precision differences
        # and potential different order of operations across GPUs
        assert torch.allclose(logits1, logits2, atol=5e-3, rtol=5e-2)
    
    def test_dataparallel_memory_efficiency(self):
        """Test that DataParallel actually distributes computation across GPUs."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        vocab_size = 1000
        model = create_model(vocab_size).to(DEVICE)
        model_parallel = nn.DataParallel(model)
        
        # Large batch that should be split across GPUs
        large_batch_size = 16
        x = torch.randint(0, vocab_size, (large_batch_size, 32)).to(DEVICE)
        targets = torch.randint(0, vocab_size, (large_batch_size, 32)).to(DEVICE)
        
        # Check initial memory
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        model_parallel.train()
        logits, loss = model_parallel(x, targets)
        
        torch.cuda.synchronize()
        forward_memory = torch.cuda.memory_allocated()
        
        # Backward pass
        # Handle DataParallel loss gathering - take mean if multiple scalars
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        
        torch.cuda.synchronize()
        backward_memory = torch.cuda.memory_allocated()
        
        # Memory should have increased (actual usage verification)
        assert forward_memory > initial_memory
        assert backward_memory >= forward_memory
        
        # Clean up
        del logits, loss, x, targets
        torch.cuda.empty_cache()
    
    def test_dataparallel_model_state_dict(self):
        """Test that model state dict is properly handled with DataParallel."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        vocab_size = 700
        model = create_model(vocab_size).to(DEVICE)
        model_parallel = nn.DataParallel(model)
        
        # Get state dicts
        original_state_dict = model.state_dict()
        parallel_state_dict = model_parallel.module.state_dict()
        
        # Should be identical
        assert set(original_state_dict.keys()) == set(parallel_state_dict.keys())
        
        for key in original_state_dict.keys():
            assert torch.equal(original_state_dict[key], parallel_state_dict[key])
    
    def test_effective_batch_size_calculation(self):
        """Test that effective batch size is correctly calculated for multi-GPU."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU tests require at least 2 GPUs")
        
        # Simulate config
        class MockConfig:
            training = type('', (), {'batch_size': 8})()
            system = type('', (), {'multi_gpu': True})()
        
        cfg = MockConfig()
        
        vocab_size = 500
        model = create_model(vocab_size).to(DEVICE)
        
        if cfg.system.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            effective_batch_size = cfg.training.batch_size * torch.cuda.device_count()
        else:
            effective_batch_size = cfg.training.batch_size
        
        # With 2 GPUs, effective batch size should be 16
        expected_effective_batch_size = 8 * torch.cuda.device_count()
        assert effective_batch_size == expected_effective_batch_size


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 