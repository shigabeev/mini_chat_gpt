# GPT-2 Training on TinyStories

A clean, minimal implementation of GPT-2 (~100M parameters) for pretraining on the TinyStories dataset. Written in the style of Andrej Karpathy's educational code with modern enhancements.

## Features

🚀 **Modern Architecture Enhancements:**
- **FlashAttention** for memory-efficient attention computation
- **RoPE** (Rotary Position Embedding) for better position encoding
- **Pre-Norm LayerNorm** for improved training stability
- **SwiGLU** activation with parallel FFN computation
- **Mixed Precision Training** (bfloat16) for faster training

⚡ **Training Features:**
- Automatic data downloading and tokenization
- Cosine learning rate schedule with warmup
- Gradient clipping and weight decay
- WandB integration for experiment tracking
- Automatic checkpointing and resuming
- Model compilation for faster inference

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd mini_chat_gpt

# Install dependencies
pip install -r requirements.txt

# Optional: Install FlashAttention for better performance
pip install flash-attn --no-build-isolation
```

### 2. Training

Start training with default hyperparameters:

```bash
python train.py
```

For multi-GPU training (automatically detects available GPUs):

```bash
python train.py system.multi_gpu=true training.batch_size=16
```

The script will automatically:
- Download TinyStories dataset (~2GB)
- Tokenize the data using tiktoken
- Create model with ~100M parameters
- Start training with mixed precision
- Use all available GPUs if multi_gpu=true

### 3. Custom Configuration

Modify `config.yaml` or override parameters:

```bash
# Train with different batch size
python train.py training.batch_size=16

# Use different learning rate
python train.py training.learning_rate=1e-4

# Resume from checkpoint
python train.py checkpointing.resume_from=./checkpoints/checkpoint_step_10000.pt
```

### 4. Text Generation

Generate text using a trained model:

```bash
python generate.py --checkpoint ./checkpoints/best_model.pt --prompt "Once upon a time"
```

## Model Architecture

The model implements a standard transformer decoder with modern enhancements:

```
GPT-2 Base Configuration:
- Vocabulary Size: ~100k (tiktoken cl100k_base)
- Model Dimension: 768
- Layers: 12
- Attention Heads: 12
- Context Length: 1024
- Parameters: ~124M
```

### Key Components

1. **RoPE Position Encoding**: Replaces learned positional embeddings
2. **Pre-Norm LayerNorm**: Applied before attention and FFN
3. **SwiGLU Activation**: `SwiGLU(x) = Swish(W1(x)) ⊙ W3(x) @ W2`
4. **FlashAttention**: Memory-efficient attention implementation
5. **Weight Tying**: Embedding and output projection weights are shared

## Performance

The implementation is optimized for training efficiency:

- **Mixed Precision**: 2x memory reduction with bfloat16
- **FlashAttention**: ~3x faster attention computation
- **Multi-GPU Support**: Linear scaling with DataParallel
- **Model Compilation**: Additional 10-20% speedup
- **Efficient Data Loading**: Memory-mapped tokenized data

Expected training time:
- Single RTX 4090: ~24 hours for 100k steps
- Dual RTX 4090: ~12 hours for 100k steps

## File Structure

```
├── model.py          # GPT model implementation
├── dataloader.py     # TinyStories dataset and dataloader
├── train.py          # Training script with Hydra
├── generate.py       # Text generation script
├── config.yaml       # Hydra configuration
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Configuration

Key hyperparameters in `config.yaml`:

```yaml
model:
  dim: 768              # Model dimension
  n_layers: 12          # Number of transformer layers
  n_heads: 12           # Number of attention heads
  max_seq_len: 1024     # Context length

training:
  batch_size: 8         # Batch size (adjust for your GPU)
  learning_rate: 3e-4   # Peak learning rate
  max_steps: 100000     # Total training steps
  warmup_steps: 2000    # LR warmup steps

system:
  multi_gpu: true       # Enable multi-GPU training
  mixed_precision: true # Use bfloat16 training
  compile: true         # Compile model for speed
```

## Memory Requirements

- **Single GPU Training**: ~10GB VRAM (batch_size=8, seq_len=1024)
- **Multi-GPU Training**: ~8GB VRAM per GPU (batch_size=16, seq_len=1024)
- **Inference**: ~1GB VRAM
- **Data**: ~4GB disk space for tokenized TinyStories

For smaller GPUs, reduce `batch_size` or `seq_len` in config.

## Tips for Training

1. **Monitor Loss**: Training loss should decrease to ~2.5-3.0
2. **Validation**: Check validation loss every 1k steps
3. **Generation**: Test generation quality during training
4. **Checkpoints**: Keep multiple checkpoints for comparison
5. **WandB**: Use WandB for detailed training monitoring

## Example Generation

After training, the model generates coherent short stories:

```
Prompt: "Once upon a time, there was a little girl named"

Generated:
"Once upon a time, there was a little girl named Lily. She loved to play 
with her toys and go on adventures. One day, she found a magic key in her 
backyard. When she used the key, it opened a door to a wonderful garden 
filled with colorful flowers and friendly animals..."
```

## Citation

```bibtex
@article{tinystories2023,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

## Acknowledgments

- Inspired by Andrej Karpathy's educational code style
- Built on the TinyStories dataset by Microsoft Research
- Uses modern techniques from recent transformer research 