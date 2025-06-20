"""
Chat generation script for SFT-trained TinyStories model.
Supports instruction-following conversations.
"""

import torch
import tiktoken
from mini_chat_gpt.model import GPT
from typing import Optional, List, Dict


def load_sft_model(checkpoint_path: str, device: str = "cuda") -> tuple[GPT, tiktoken.Encoding]:
    """Load SFT-trained model from checkpoint."""
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


def format_chat_prompt(user_message: str, system_message: str = None) -> str:
    """Format user message into chat format."""
    if system_message is None:
        system_message = "You are a helpful assistant that answers questions about stories."
    
    return f"<|system|>{system_message}<|user|>{user_message}<|assistant|>"


def generate_chat_response(
    model: GPT,
    tokenizer: tiktoken.Encoding,
    user_message: str,
    system_message: str = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cuda"
) -> str:
    """Generate response to user message in chat format."""
    
    # Format prompt
    chat_prompt = format_chat_prompt(user_message, system_message)
    
    # Encode prompt (allow special tokens)
    tokens = tokenizer.encode(chat_prompt, allowed_special={"<|endoftext|>"})
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate response
    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_tokens=[tokenizer.encode('<|endoftext|>', allowed_special={"<|endoftext|>"})[0] if tokenizer.encode('<|endoftext|>', allowed_special={"<|endoftext|>"}) else 50256]
        )
    
    # Decode full generated text
    generated_tokens = generated[0].tolist()
    full_text = tokenizer.decode(generated_tokens)
    
    # Extract only the assistant's response
    assistant_start = full_text.find('<|assistant|>')
    if assistant_start != -1:
        assistant_start += len('<|assistant|>')
        # Find end of response
        end_markers = ['<|endoftext|>', '<|user|>', '<|system|>']
        assistant_end = len(full_text)
        for marker in end_markers:
            pos = full_text.find(marker, assistant_start)
            if pos != -1:
                assistant_end = min(assistant_end, pos)
        
        response = full_text[assistant_start:assistant_end].strip()
        return response
    else:
        return full_text


def interactive_chat(model: GPT, tokenizer: tiktoken.Encoding, device: str = "cuda"):
    """Run interactive chat session with SFT model."""
    print("=== TinyStories SFT Chat ===")
    print("Type your questions or 'quit' to exit")
    print("Examples:")
    print("- 'Please summarize this story: [story text]'")
    print("- 'Write a story using these words: cat, tree, happy'")
    print("- 'What is the moral of this story: [story text]'")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("Assistant: ", end="", flush=True)
        
        try:
            response = generate_chat_response(
                model, tokenizer, user_input, device=device
            )
            print(response)
        except Exception as e:
            print(f"Error generating response: {e}")


def test_sft_capabilities(model: GPT, tokenizer: tiktoken.Encoding, device: str = "cuda"):
    """Test different SFT capabilities."""
    print("=== Testing SFT Capabilities ===\n")
    
    # Test story
    test_story = """Sara and Ben were playing in the park. They liked to climb the big oak tree and pretend they were birds. They made nests with leaves and twigs and sang songs. But today, the sky was gloomy and the wind was cold. Sara felt sad and cold. She wanted to go home and have some hot cocoa. "Ben, I want to quit," she said. "It's too cold and dark. Let's go home." Ben looked at Sara and frowned. He liked the oak tree and the park. He wanted to stay and play. "No, Sara, don't quit," he said. "It's fun here. Look, there's a squirrel. Let's chase it." Sara shook her head. She didn't want to chase the squirrel. She wanted to go home and have some hot cocoa. "Please, Ben, let's go home," she said. "We can play here another day. I'm cold and hungry." Ben saw that Sara was shivering and looked unhappy. He loved his sister and didn't want her to be sad. He nodded and smiled. "Okay, Sara, let's go home," he said. "We can have some hot cocoa and cookies. And we can play with our toys." Sara hugged Ben and thanked him. They climbed down the oak tree and ran to their mom, who was waiting for them."""
    
    test_cases = [
        ("Summarization", f"Please provide a brief summary of the following story:\n\n{test_story}"),
        ("Story Generation", "Write a short story that uses all of these words: cat, tree, happy"),
        ("Q&A", f"Based on this story, who is the main character?\n\nStory: {test_story}"),
        ("Character Question", f"Based on this story, what did Sara want to do?\n\nStory: {test_story}"),
    ]
    
    for task_name, prompt in test_cases:
        print(f"🔍 Testing {task_name}:")
        print(f"Prompt: {prompt[:100]}...")
        print("Response: ", end="", flush=True)
        
        try:
            response = generate_chat_response(
                model, tokenizer, prompt, 
                max_new_tokens=150, 
                temperature=0.7,
                device=device
            )
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text using SFT-trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to SFT checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat")
    parser.add_argument("--test", action="store_true", help="Run capability tests")
    parser.add_argument("--prompt", help="Single prompt to test")
    
    args = parser.parse_args()
    
    print(f"Loading SFT model from {args.checkpoint}...")
    model, tokenizer = load_sft_model(args.checkpoint, args.device)
    print(f"Model loaded! Parameters: {model.get_num_params():,}")
    
    if args.interactive:
        interactive_chat(model, tokenizer, args.device)
    elif args.test:
        test_sft_capabilities(model, tokenizer, args.device)
    elif args.prompt:
        print(f"Prompt: {args.prompt}")
        print("Response: ", end="", flush=True)
        response = generate_chat_response(model, tokenizer, args.prompt, device=args.device)
        print(response)
    else:
        print("Please specify --interactive, --test, or --prompt")


if __name__ == "__main__":
    main() 