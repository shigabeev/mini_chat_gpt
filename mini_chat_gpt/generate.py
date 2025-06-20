"""
Text generation script using trained GPT model.
"""

import torch
import tiktoken
from mini_chat_gpt.model import GPT
from typing import Optional


def load_model(checkpoint_path: str, device: str = "cuda") -> tuple[GPT, tiktoken.Encoding]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Create model
    model = GPT(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        max_seq_len=config['max_seq_len'],
    )
    
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
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    return model, tokenizer


def generate_text(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    stop_tokens: Optional[list] = ['<|endoftext|>'],
    device: str = "cuda"
) -> str:
    """Generate text from prompt."""
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Convert stop tokens from strings to token IDs if provided
    stop_token_ids = None
    if stop_tokens is not None:
        stop_token_ids = []
        for stop_token in stop_tokens:
            # Handle both string and already-encoded stop tokens
            if isinstance(stop_token, str):
                encoded = tokenizer.encode(stop_token)
                if encoded:  # Only add if tokenizer returns non-empty result
                    stop_token_ids.extend(encoded)
            elif isinstance(stop_token, int):
                stop_token_ids.append(stop_token)
        stop_token_ids = stop_token_ids if stop_token_ids else None
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=stop_token_ids
        )
    
    # Decode
    generated_tokens = generated[0].tolist()
    text = tokenizer.decode(generated_tokens)
    
    return text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--stop_tokens", type=str, nargs="*", help="Stop tokens (e.g., --stop_tokens '<|endoftext|>' '\\n\\n')")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer = load_model(args.checkpoint, args.device)
    
    print(f"Generating text with prompt: '{args.prompt}'")
    if args.stop_tokens:
        print(f"Stop tokens: {args.stop_tokens}")
    
    generated_text = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_tokens=args.stop_tokens,
        device=args.device
    )
    
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(generated_text)
    print("="*50) 