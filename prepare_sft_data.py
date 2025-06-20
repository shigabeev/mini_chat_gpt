#!/usr/bin/env python3
"""
Data preparation script for TinyStories-Instruct SFT.
Downloads and prepares the instruction dataset.
"""

import os
import requests
import gzip
from pathlib import Path


def download_file(url: str, filepath: str, description: str = ""):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)
    
    print(f"\n✅ Downloaded {filepath}")


def create_sample_data():
    """Create sample data for testing if full dataset is not available."""
    print("Creating sample TinyStories-Instruct data for testing...")
    
    sample_data = """Features: Dialogue
Words: quit, oak, gloomy
Summary: Sara and Ben were playing in the park, but Sara wanted to go home because it was cold and dark. Ben convinced her to stay and play, but eventually agreed to go home and have hot cocoa.
Story: 

Sara and Ben were playing in the park. They liked to climb the big oak tree and pretend they were birds. They made nests with leaves and twigs and sang songs.
But today, the sky was gloomy and the wind was cold. Sara felt sad and cold. She wanted to go home and have some hot cocoa.
"Ben, I want to quit," she said. "It's too cold and dark. Let's go home."
Ben looked at Sara and frowned. He liked the oak tree and the park. He wanted to stay and play.
"No, Sara, don't quit," he said. "It's fun here. Look, there's a squirrel. Let's chase it."
Sara shook her head. She didn't want to chase the squirrel. She wanted to go home and have some hot cocoa.
"Please, Ben, let's go home," she said. "We can play here another day. I'm cold and hungry."
Ben saw that Sara was shivering and looked unhappy. He loved his sister and didn't want her to be sad. He nodded and smiled.
"Okay, Sara, let's go home," he said. "We can have some hot cocoa and cookies. And we can play with our toys."
Sara hugged Ben and thanked him. They climbed down the oak tree and ran to their mom, who was waiting for them. They told her about their adventure and asked for some hot cocoa and cookies. Mom smiled and agreed. She was proud of her children for being brave and kind. They went home and had a cozy and happy time.
<|endoftext|>
Summary: Lily steals a new bike from a store and gets into an accident while riding it, resulting in her getting hurt and being sent to jail, losing her parents' trust and love. The moral of the story is not to steal.
Words: ride, work, upset
Features: Dialogue, BadEnding, MoralValue
Story: 

Lily liked to ride her bike. She rode it every day after work. Work was a place where she helped her mom and dad with chores. She liked work, but she liked riding her bike more.
One day, she saw a new bike in the store. It was shiny and red and had a bell. Lily wanted the new bike very much. She asked her mom and dad if they could buy it for her. They said no. They said the new bike was too expensive and that she already had a good bike.
Lily was upset. She did not listen to her mom and dad. She thought they were mean and unfair. She decided to take the new bike without paying. She waited until the store was busy and then she sneaked out with the bike.
She rode the new bike very fast. She felt happy and proud. She rang the bell and smiled. She did not see the car coming. The car hit her and the bike. Lily and the bike flew in the air and then crashed on the ground.
Lily was hurt very badly. She cried and screamed. The store owner and her mom and dad came running. They saw what she had done. They were angry and sad. They called an ambulance and the police. Lily had to go to the hospital and then to jail. She lost her old bike and her new bike. She also lost her mom and dad's trust and love.
The moral of the story is: do not steal. Stealing is wrong and dangerous. It can hurt you and others. Be happy with what you have and listen to your mom and dad. They know what is best for you.
<|endoftext|>
Features: Dialogue
Words: happy, cat, tree
Summary: Tom finds a sad cat stuck in a tree and helps it down, making both of them happy and becoming friends.
Story: 

Tom was walking in the park when he heard a sad sound. He looked up and saw a cat stuck high in a tree. The cat was crying because it was scared and couldn't get down.
"Don't worry, little cat," Tom said. "I will help you."
Tom was good at climbing trees. He climbed up carefully and gently picked up the cat. The cat was soft and warm.
"There you go," Tom said as he climbed down with the cat. "You're safe now."
The cat was so happy! It purred and rubbed against Tom's leg. Tom was happy too because he helped the cat.
From that day on, the cat and Tom were best friends. They played in the park every day under the big tree.
<|endoftext|>"""

    # Create TinyStories directory
    os.makedirs("TinyStories", exist_ok=True)
    
    # Write sample training data (repeat the sample multiple times for training)
    with open("TinyStories/TinyStoriesInstruct-train.txt", "w", encoding='utf-8') as f:
        for i in range(100):  # Repeat 100 times for a larger training set
            f.write(sample_data)
            if i < 99:
                f.write("\n")
    
    # Write sample validation data (just one copy for validation)
    with open("TinyStories/TinyStoriesInstruct-valid.txt", "w", encoding='utf-8') as f:
        f.write(sample_data)
    
    print("✅ Created sample TinyStories-Instruct data:")
    print("   - TinyStories/TinyStoriesInstruct-train.txt")
    print("   - TinyStories/TinyStoriesInstruct-valid.txt")


def download_tinystories_instruct():
    """Download the actual TinyStories-Instruct dataset."""
    os.makedirs("TinyStories", exist_ok=True)
    
    # Note: These URLs are hypothetical - replace with actual URLs when available
    urls = {
        "TinyStoriesInstruct-train.txt": "https://example.com/TinyStoriesInstruct-train.txt",
        "TinyStoriesInstruct-valid.txt": "https://example.com/TinyStoriesInstruct-valid.txt",
    }
    
    for filename, url in urls.items():
        filepath = f"TinyStories/{filename}"
        if os.path.exists(filepath):
            print(f"✅ {filename} already exists")
            continue
        
        try:
            download_file(url, filepath, filename)
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            print("   Creating sample data instead...")
            return False
    
    return True


def test_data_loading():
    """Test the SFT dataloader with the prepared data."""
    print("\n🧪 Testing SFT dataloader...")
    
    try:
        from mini_chat_gpt.sft_dataloader import create_sft_dataloader
        
        # Test loading
        train_loader, vocab_size = create_sft_dataloader(
            "./TinyStories", 
            batch_size=2, 
            seq_len=512, 
            train=True,
            num_workers=0
        )
        
        print(f"✅ Dataloader created successfully!")
        print(f"   - Vocab size: {vocab_size}")
        print(f"   - Training samples: {len(train_loader.dataset)}")
        
        # Test a batch
        input_ids, labels = next(iter(train_loader))
        print(f"   - Batch shape: {input_ids.shape}")
        print(f"   - Label shape: {labels.shape}")
        
        # Show training tokens vs masked tokens
        train_tokens = (labels != -100).sum().item()
        total_tokens = labels.numel()
        print(f"   - Training tokens: {train_tokens}/{total_tokens} ({train_tokens/total_tokens*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error testing dataloader: {e}")


def main():
    print("🚀 TinyStories-Instruct Data Preparation")
    print("=" * 50)
    
    # Check if data already exists
    train_file = "TinyStories/TinyStoriesInstruct-train.txt"
    valid_file = "TinyStories/TinyStoriesInstruct-valid.txt"
    
    if os.path.exists(train_file) and os.path.exists(valid_file):
        print("✅ TinyStories-Instruct data already exists!")
    else:
        print("📥 TinyStories-Instruct data not found.")
        
        # Try to download real data, fallback to sample data
        if not download_tinystories_instruct():
            create_sample_data()
    
    # Test the dataloader
    test_data_loading()
    
    print("\n🎯 Next Steps:")
    print("1. Ensure you have a pretrained checkpoint")
    print("2. Update the checkpoint path in sft_config.yaml")
    print("3. Start SFT training:")
    print("   python mini_chat_gpt/sft_train.py")
    print("4. Test your SFT model:")
    print("   python mini_chat_gpt/sft_generate.py --checkpoint /path/to/sft_checkpoint.pt --test")


if __name__ == "__main__":
    main() 