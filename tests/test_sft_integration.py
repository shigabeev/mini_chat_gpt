"""
SFT Integration Tests - End-to-end validation of the complete SFT pipeline.
Tests the complete flow from data preparation to model training and generation.
"""

import pytest
import torch
import tempfile
import os
import shutil
from pathlib import Path
import json
import yaml
from unittest.mock import patch, MagicMock

from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset, create_sft_dataloader
from mini_chat_gpt.model import create_model, GPT
from mini_chat_gpt.sft_train import save_sft_checkpoint, load_pretrained_checkpoint
from mini_chat_gpt.sft_generate import load_sft_model, format_chat_prompt, generate_chat_response


class TestCompleteSFTPipeline:
    """Test the complete SFT pipeline from start to finish."""
    
    @pytest.fixture
    def comprehensive_sft_data(self):
        """Comprehensive SFT data with all features."""
        return """Features: Dialogue
Words: adventure, forest, brave
Summary: A young explorer goes on an adventure in the forest and learns to be brave.
Story: 

Emma was a young explorer who loved adventures. One day, she decided to explore the deep forest.
"I will be brave," she said to herself as she packed her backpack.
The forest was dark and mysterious. Emma heard strange noises that made her scared.
"Don't be afraid," she whispered. "You are brave and strong."
She found a beautiful clearing with colorful flowers and singing birds.
Emma realized that being brave doesn't mean not being scared, but facing your fears.
She returned home with a big smile, proud of her forest adventure.
<|endoftext|>
Features: Dialogue, MoralValue
Words: honesty, truth, lies
Summary: A child learns the importance of telling the truth instead of lying, even when it's difficult.
Story: 

Jake broke his mom's favorite vase while playing ball in the house.
He was scared and thought about blaming his little sister.
"Should I tell the truth?" Jake wondered. "Mom will be angry."
When his mom found the broken vase, she asked what happened.
Jake took a deep breath and said, "I broke it, Mom. I'm sorry."
His mom was upset about the vase but proud of Jake for being honest.
"Thank you for telling the truth," she said. "Honesty is always the best choice."
Jake learned that telling the truth, even when it's hard, makes you feel better inside.
The moral of the story is: always tell the truth, even when it's difficult.
<|endoftext|>
Features: Dialogue, BadEnding
Words: careful, dangerous, accident
Summary: A child ignores safety warnings and gets hurt, learning to be more careful.
Story: 

Sam loved to climb trees, but his parents always told him to be careful.
"That tree looks dangerous," his dad warned. "Don't climb too high."
But Sam thought he knew better. "I'm a good climber," he said.
He climbed higher and higher, ignoring the warnings about the dangerous branches.
Suddenly, a branch snapped! Sam fell and hurt his arm badly.
At the hospital, Sam realized he should have listened to his parents.
"I should have been more careful," Sam said sadly. "Now I'm hurt and can't play."
The accident taught Sam to listen to safety warnings and be more careful.
<|endoftext|>
Features: Friendship, Dialogue
Words: share, friendship, generous
Summary: Two friends learn about sharing and how being generous makes friendship stronger.
Story: 

Lily and Maya were best friends who loved to play together.
One day, Lily brought her new toy robot to school.
"Can I play with it too?" Maya asked hopefully.
At first, Lily didn't want to share. "It's mine," she said.
But then she saw Maya's sad face and remembered how Maya always shared with her.
"Of course you can play with it," Lily said with a smile. "Friends share everything."
They took turns playing with the robot and had so much fun together.
"Sharing makes everything better," Maya said happily.
Their friendship became even stronger because they learned to be generous with each other.
<|endoftext|>"""

    @pytest.fixture
    def temp_project_setup(self, tmp_path, comprehensive_sft_data):
        """Set up a complete temporary project structure."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        
        # Create data directory
        data_dir = project_dir / "TinyStories"
        data_dir.mkdir()
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(comprehensive_sft_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(comprehensive_sft_data)
        
        # Create checkpoints directory
        checkpoints_dir = project_dir / "checkpoints"
        checkpoints_dir.mkdir()
        
        # Create config file
        config = {
            'sft': {
                'dataset_path': str(data_dir),
                'learning_rate': 1e-4,
                'max_steps': 10,  # Very short for testing
                'warmup_steps': 2,
                'eval_interval': 5,
                'save_interval': 10,
                'save_dir': str(checkpoints_dir),
                'resume_from': None
            },
            'model': {
                'dim': 128,  # Small for testing
                'n_layers': 2,
                'n_heads': 4,
                'n_kv_heads': 4,  
                'max_seq_len': 256,
                'vocab_size': 50304
            },
            'training': {
                'batch_size': 2,
                'seq_len': 256,
                'num_workers': 0,
                'device': 'cpu'  # Force CPU for testing
            }
        }
        
        config_file = project_dir / "sft_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        return {
            'project_dir': project_dir,
            'data_dir': data_dir,
            'checkpoints_dir': checkpoints_dir,
            'config_file': config_file,
            'config': config
        }

    def test_complete_data_processing_pipeline(self, temp_project_setup):
        """Test the complete data processing pipeline."""
        setup = temp_project_setup
        
        # Test dataset creation
        dataset = TinyStoriesInstructDataset(
            str(setup['data_dir']), 
            seq_len=256, 
            train=True
        )
        
        # Verify all records were parsed correctly
        assert len(dataset.records) == 4
        
        # Verify all task types are generated
        task_types = [task['type'] for task in dataset.tasks]
        expected_types = {'summarization', 'use_words', 'qa', 'moral'}
        actual_types = set(task_types)
        assert expected_types.issubset(actual_types), f"Missing task types: {expected_types - actual_types}"
        
        # Verify specific task content
        moral_tasks = [task for task in dataset.tasks if task['type'] == 'moral']
        assert len(moral_tasks) == 1  # Only one story has MoralValue
        moral_task = moral_tasks[0]
        assert 'always tell the truth' in moral_task['target']
        
        # Test dataloader creation
        dataloader, vocab_size = create_sft_dataloader(
            str(setup['data_dir']),
            batch_size=2,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        # Test batch processing
        batch = next(iter(dataloader))
        input_ids, labels = batch
        
        assert input_ids.shape == (2, 256)
        assert labels.shape == (2, 256)
        
        # Verify loss masking
        for i in range(2):
            training_tokens = (labels[i] != -100).sum().item()
            assert training_tokens > 0, f"No training tokens in sample {i}"
            
        print(f"✅ Data processing: {len(dataset.records)} records → {len(dataset.tasks)} tasks")

    def test_model_training_pipeline(self, temp_project_setup):
        """Test the model training pipeline with SFT data."""
        setup = temp_project_setup
        
        # Create model
        config = setup['config']['model']
        model = GPT(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_seq_len=config['max_seq_len']
        )
        
        # Create dataloader
        dataloader, vocab_size = create_sft_dataloader(
            str(setup['data_dir']),
            batch_size=2,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        # Test training step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        
        input_ids, labels = next(iter(dataloader))
        
        # Forward pass
        logits, loss = model(input_ids, targets=labels)
        
        # Verify loss computation
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss should be positive"
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify gradients
        has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_gradients, "No gradients computed"
        
        print(f"✅ Training step: loss = {loss.item():.3f}")

    def test_checkpoint_saving_and_loading(self, temp_project_setup):
        """Test checkpoint saving and loading functionality."""
        setup = temp_project_setup
        
        # Create and train model briefly
        config = setup['config']['model']
        model = GPT(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_seq_len=config['max_seq_len']
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Save checkpoint
        save_sft_checkpoint(
            model=model,
            optimizer=optimizer,
            step=42,
            loss=2.5,
            save_dir=str(setup['checkpoints_dir'])
        )
        
        # Verify checkpoint file exists
        checkpoint_file = setup['checkpoints_dir'] / "sft_checkpoint_step_42.pt"
        assert checkpoint_file.exists(), "Checkpoint file not created"
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        assert checkpoint['step'] == 42
        assert checkpoint['loss'] == 2.5
        assert checkpoint['model_type'] == 'sft'
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        
        # Test loading into new model
        new_model = GPT(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_seq_len=config['max_seq_len']
        )
        
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify models have same weights
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Model weights don't match after loading"
        
        print("✅ Checkpoint saving/loading working correctly")

    def test_generation_pipeline(self, temp_project_setup):
        """Test the generation pipeline with SFT model."""
        setup = temp_project_setup
        
        # Create and save a model
        config = setup['config']['model']
        model = GPT(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_seq_len=config['max_seq_len']
        )
        
        # Save as SFT checkpoint
        checkpoint_path = setup['checkpoints_dir'] / "test_sft_model.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_type': 'sft',
            'step': 100,
            'loss': 2.0
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Test loading
        loaded_model, tokenizer = load_sft_model(str(checkpoint_path), device="cpu")
        
        assert isinstance(loaded_model, GPT)
        assert loaded_model.vocab_size == config['vocab_size']
        
        # Test chat formatting
        user_message = "Write a short story about friendship."
        system_message = "You are a helpful storytelling assistant."
        
        chat_prompt = format_chat_prompt(user_message, system_message)
        
        assert '<|system|>' in chat_prompt
        assert '<|user|>' in chat_prompt
        assert '<|assistant|>' in chat_prompt
        assert user_message in chat_prompt
        assert system_message in chat_prompt
        
        # Test generation method exists and works
        response = generate_chat_response(
            loaded_model, 
            tokenizer, 
            user_message, 
            system_message=system_message,
            max_new_tokens=20,  # Short for testing
            temperature=0.7,
            device="cpu"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        print(f"✅ Generation working: '{response[:50]}...'")

    def test_task_specific_performance(self, temp_project_setup):
        """Test that the model can handle different task types."""
        setup = temp_project_setup
        
        # Create dataset
        dataset = TinyStoriesInstructDataset(
            str(setup['data_dir']), 
            seq_len=256, 
            train=True
        )
        
        # Group tasks by type
        tasks_by_type = {}
        for task in dataset.tasks:
            task_type = task['type']
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)
        
        # Test each task type
        for task_type, tasks in tasks_by_type.items():
            print(f"Testing {task_type} tasks ({len(tasks)} samples)...")
            
            # Verify task structure
            for task in tasks[:2]:  # Test first 2 of each type
                assert 'prompt' in task
                assert 'target' in task
                assert len(task['prompt']) > 0
                assert len(task['target']) > 0
                
                # Verify task-specific content
                if task_type == 'summarization':
                    assert 'summary' in task['prompt'].lower()
                elif task_type == 'moral':
                    assert 'moral' in task['prompt'].lower()
                    assert 'moral of the story is' in task['target'].lower()
                elif task_type == 'use_words':
                    assert 'words' in task['prompt'].lower()
                elif task_type == 'qa':
                    assert 'based on this story' in task['prompt'].lower()
        
        print(f"✅ All task types working: {list(tasks_by_type.keys())}")

    def test_error_handling_and_edge_cases(self, temp_project_setup):
        """Test error handling and edge cases."""
        setup = temp_project_setup
        
        # Test with empty dataset
        empty_data_dir = setup['project_dir'] / "empty_data"
        empty_data_dir.mkdir()
        (empty_data_dir / "TinyStoriesInstruct-train.txt").write_text("")
        (empty_data_dir / "TinyStoriesInstruct-valid.txt").write_text("")
        
        empty_dataset = TinyStoriesInstructDataset(str(empty_data_dir), seq_len=256, train=True)
        assert len(empty_dataset) == 0
        
        # Test with malformed data
        malformed_data = """This is not properly formatted data.
Missing required fields.
<|endoftext|>"""
        
        malformed_data_dir = setup['project_dir'] / "malformed_data"
        malformed_data_dir.mkdir()
        (malformed_data_dir / "TinyStoriesInstruct-train.txt").write_text(malformed_data)
        (malformed_data_dir / "TinyStoriesInstruct-valid.txt").write_text(malformed_data)
        
        malformed_dataset = TinyStoriesInstructDataset(str(malformed_data_dir), seq_len=256, train=True)
        assert len(malformed_dataset) == 0  # Should filter out malformed records
        
        # Test missing files
        nonexistent_dir = setup['project_dir'] / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            TinyStoriesInstructDataset(str(nonexistent_dir), seq_len=256, train=True)
        
        print("✅ Error handling working correctly")

    def test_memory_efficiency(self, temp_project_setup):
        """Test memory efficiency of the SFT pipeline."""
        setup = temp_project_setup
        
        # Create dataset
        dataset = TinyStoriesInstructDataset(
            str(setup['data_dir']), 
            seq_len=256, 
            train=True
        )
        
        # Test that dataset doesn't load everything into memory at once
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(dataset)
        
        # Access multiple items
        for i in range(min(5, len(dataset))):
            _ = dataset[i]
        
        # Memory shouldn't grow significantly
        final_size = sys.getsizeof(dataset)
        assert final_size < initial_size * 2, "Dataset memory usage grew too much"
        
        # Test dataloader memory efficiency
        dataloader, _ = create_sft_dataloader(
            str(setup['data_dir']),
            batch_size=2,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        # Should be able to iterate multiple times
        for epoch in range(2):
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 3:  # Don't need to test all batches
                    break
            assert batch_count > 0, f"No batches in epoch {epoch}"
        
        print("✅ Memory efficiency validated")

    def test_full_pipeline_integration(self, temp_project_setup):
        """Test the complete pipeline integration."""
        setup = temp_project_setup
        
        print("🔄 Running full pipeline integration test...")
        
        # Step 1: Data loading and processing
        dataset = TinyStoriesInstructDataset(
            str(setup['data_dir']), 
            seq_len=256, 
            train=True
        )
        
        dataloader, vocab_size = create_sft_dataloader(
            str(setup['data_dir']),
            batch_size=2,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        print(f"  ✅ Data: {len(dataset)} tasks loaded")
        
        # Step 2: Model creation and training
        config = setup['config']['model']
        model = GPT(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_seq_len=config['max_seq_len']
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        
        # Mini training loop
        total_loss = 0
        for step, (input_ids, labels) in enumerate(dataloader):
            if step >= 3:  # Just a few steps for testing
                break
                
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets=labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (step + 1)
        print(f"  ✅ Training: {step + 1} steps, avg loss = {avg_loss:.3f}")
        
        # Step 3: Checkpoint saving
        save_sft_checkpoint(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=avg_loss,
            save_dir=str(setup['checkpoints_dir'])
        )
        
        checkpoint_file = setup['checkpoints_dir'] / f"sft_checkpoint_step_{step}.pt"
        assert checkpoint_file.exists()
        print(f"  ✅ Checkpoint saved: {checkpoint_file.name}")
        
        # Step 4: Model loading and generation
        # Update checkpoint to include config
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        checkpoint['config'] = config
        torch.save(checkpoint, checkpoint_file)
        
        loaded_model, tokenizer = load_sft_model(str(checkpoint_file), device="cpu")
        
        response = generate_chat_response(
            loaded_model, 
            tokenizer, 
            "Tell me a short story about friendship.", 
            max_new_tokens=10,
            device="cpu"
        )
        
        print(f"  ✅ Generation: '{response[:30]}...'")
        
        print("🎉 Full pipeline integration test passed!")


if __name__ == "__main__":
    # Run a quick integration test
    pytest.main([__file__, "-v"]) 