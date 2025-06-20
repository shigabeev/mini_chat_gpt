#!/usr/bin/env python3
"""
SFT Test Runner - Validates the complete SFT implementation.
Runs tests and provides detailed feedback on functionality.
"""

import sys
import subprocess
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_basic_validation():
    """Run basic validation tests without pytest."""
    print("🔍 Running basic SFT validation...")
    
    try:
        # Test 1: Import all SFT modules
        print("  ✓ Testing imports...")
        from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset, create_sft_dataloader
        from mini_chat_gpt.sft_train import save_sft_checkpoint, load_pretrained_checkpoint
        from mini_chat_gpt.sft_generate import load_sft_model, format_chat_prompt, generate_chat_response
        print("    ✅ All SFT modules imported successfully")
        
        # Test 2: Create sample data and test dataloader
        print("  ✓ Testing SFT dataloader...")
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create sample data
            sample_data = """Features: Dialogue
Words: test, validation, working
Summary: A test story to validate the SFT dataloader is working correctly.
Story: 

This is a test story for validation. It has dialogue and structure.
"Hello," said the character. "This test should work!"
The character was happy when the test passed.
<|endoftext|>
Features: Dialogue, MoralValue
Words: share, kind, lesson
Summary: A story about sharing and being kind to others.
Story: 

Tom had many toys but didn't want to share. His friend was sad.
"Sharing is caring," said his mom. Tom learned to share his toys.
Everyone was happy when they played together.
The moral of the story is: sharing makes everyone happy.
<|endoftext|>"""
            
            data_dir = Path(tmp_dir) / "TinyStories"
            data_dir.mkdir()
            (data_dir / "TinyStoriesInstruct-train.txt").write_text(sample_data)
            (data_dir / "TinyStoriesInstruct-valid.txt").write_text(sample_data)
            
            # Test dataset creation
            dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=256, train=True)
            assert len(dataset.records) == 2, f"Expected 2 records, got {len(dataset.records)}"
            assert len(dataset.tasks) > 2, f"Expected > 2 tasks, got {len(dataset.tasks)}"
            print(f"    ✅ Dataset created: {len(dataset.records)} records, {len(dataset.tasks)} tasks")
            
            # Test task types
            task_types = [task['type'] for task in dataset.tasks]
            expected_types = ['summarization', 'use_words', 'qa', 'moral']
            for expected_type in expected_types:
                assert expected_type in task_types, f"Missing task type: {expected_type}"
            print(f"    ✅ All task types generated: {set(task_types)}")
            
            # Test dataloader
            dataloader, vocab_size = create_sft_dataloader(
                str(data_dir), batch_size=2, seq_len=256, train=True, num_workers=0
            )
            input_ids, labels = next(iter(dataloader))
            assert input_ids.shape == (2, 256), f"Wrong input shape: {input_ids.shape}"
            assert labels.shape == (2, 256), f"Wrong label shape: {labels.shape}"
            print(f"    ✅ Dataloader working: batch shape {input_ids.shape}, vocab_size {vocab_size}")
            
            # Test loss masking
            training_tokens = (labels != -100).sum().item()
            total_tokens = labels.numel()
            training_ratio = training_tokens / total_tokens
            assert 0.02 < training_ratio < 0.5, f"Loss masking ratio seems wrong: {training_ratio:.3f}"
            print(f"    ✅ Loss masking working: {training_ratio:.1%} tokens for training")
        
        # Test 3: Model integration
        print("  ✓ Testing model integration...")
        from mini_chat_gpt.model import create_model
        import torch
        import torch.nn.functional as F
        
        # Create small model for testing
        model = create_model(vocab_size=vocab_size)
        
        # Test forward pass with SFT data
        with torch.no_grad():
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
        assert not torch.isnan(loss), "Loss computation failed"
        assert loss.item() > 0, "Loss should be positive"
        print(f"    ✅ Model forward pass working: loss = {loss.item():.3f}")
        
        # Test 4: Chat formatting
        print("  ✓ Testing chat formatting...")
        prompt = format_chat_prompt("Write a story", "You are helpful")
        assert '<|system|>' in prompt, "Missing system token"
        assert '<|user|>' in prompt, "Missing user token"
        assert '<|assistant|>' in prompt, "Missing assistant token"
        print("    ✅ Chat formatting working")
        
        # Test 5: Checkpoint functionality
        print("  ✓ Testing checkpoint functionality...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            save_sft_checkpoint(model, optimizer, step=100, loss=2.5, save_dir=tmp_dir)
            
            checkpoint_file = Path(tmp_dir) / "sft_checkpoint_step_100.pt"
            assert checkpoint_file.exists(), "Checkpoint file not created"
            
            checkpoint = torch.load(checkpoint_file, weights_only=False)
            assert checkpoint['step'] == 100, "Wrong step in checkpoint"
            assert checkpoint['model_type'] == 'sft', "Wrong model type in checkpoint"
            print("    ✅ Checkpoint saving/loading working")
        
        print("\n✅ All basic validation tests passed! SFT implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        print("Error details:")
        traceback.print_exc()
        return False


def run_pytest_tests():
    """Run the full pytest suite."""
    print("\n🧪 Running full pytest test suite...")
    
    try:
        # Run pytest on SFT tests
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_sft_functionality.py', 
            '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=project_root)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ All pytest tests passed!")
            return True
        else:
            print(f"❌ Some pytest tests failed (exit code: {result.returncode})")
            return False
            
    except FileNotFoundError:
        print("⚠️  pytest not found. Install with: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False


def run_data_preparation_test():
    """Test the data preparation script."""
    print("\n📋 Testing data preparation script...")
    
    try:
        result = subprocess.run([
            sys.executable, 'prepare_sft_data.py'
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ Data preparation script working")
            return True
        else:
            print(f"❌ Data preparation failed (exit code: {result.returncode})")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing data preparation: {e}")
        return False


def main():
    """Run all SFT tests and validation."""
    print("🚀 SFT Implementation Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Run basic validation
    if not run_basic_validation():
        all_passed = False
    
    # Run data preparation test
    if not run_data_preparation_test():
        all_passed = False
    
    # Run full pytest suite
    if not run_pytest_tests():
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! SFT implementation is ready to use.")
        print("\nNext steps:")
        print("1. Ensure you have a pretrained checkpoint")
        print("2. Update sft_config.yaml with your checkpoint path")
        print("3. Run: python mini_chat_gpt/sft_train.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 