"""
Comprehensive tests for SFT functionality.
Tests dataloader, training components, and generation.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import tiktoken
from unittest.mock import patch, MagicMock

from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset, create_sft_dataloader
from mini_chat_gpt.model import GPT, create_model
from mini_chat_gpt.sft_generate import load_sft_model, format_chat_prompt, generate_chat_response


class TestSFTDataloader:
    """Test SFT dataloader functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample TinyStories-Instruct data for testing."""
        return """Features: Dialogue
Words: quit, oak, gloomy
Summary: Sara and Ben were playing in the park, but Sara wanted to go home because it was cold and dark.
Story: 

Sara and Ben were playing in the park. They liked to climb the big oak tree and pretend they were birds.
But today, the sky was gloomy and the wind was cold. Sara felt sad and cold. She wanted to go home.
"Ben, I want to quit," she said. "It's too cold and dark. Let's go home."
Ben looked at Sara and frowned. He liked the oak tree and the park. He wanted to stay and play.
"No, Sara, don't quit," he said. "It's fun here. Look, there's a squirrel. Let's chase it."
Sara shook her head. She didn't want to chase the squirrel. She wanted to go home.
Ben saw that Sara was shivering and looked unhappy. He loved his sister and didn't want her to be sad.
"Okay, Sara, let's go home," he said. "We can have some hot cocoa and cookies."
<|endoftext|>
Summary: Lily steals a bike and gets into an accident, learning not to steal.
Words: ride, work, upset
Features: Dialogue, BadEnding, MoralValue
Story: 

Lily liked to ride her bike. She rode it every day after work.
One day, she saw a new bike in the store. It was shiny and red. Lily wanted it very much.
She asked her mom and dad if they could buy it for her. They said no.
Lily was upset. She decided to take the new bike without paying.
She rode the new bike very fast but crashed into a car. Lily was hurt very badly.
The moral of the story is: do not steal. Stealing is wrong and dangerous.
<|endoftext|>"""

    @pytest.fixture
    def temp_data_dir(self, tmp_path, sample_data):
        """Create temporary data directory."""
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(sample_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(sample_data)
        
        return str(data_dir)

    def test_dataset_initialization(self, temp_data_dir):
        """Test dataset initializes correctly."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        assert dataset.seq_len == 512
        assert dataset.train == True
        assert len(dataset.records) == 2  # Two stories in sample data
        assert len(dataset.tasks) > 2  # Should generate multiple tasks per story
        assert dataset.tokenizer.n_vocab > 0

    def test_record_parsing(self, temp_data_dir):
        """Test parsing of TinyStories-Instruct format."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        # First record
        record = dataset.records[0]
        assert record['features'] == 'Dialogue'
        assert record['words'] == 'quit, oak, gloomy'
        assert 'Sara and Ben' in record['summary']
        assert 'Sara and Ben were playing in the park' in record['story']
        
        # Second record with moral
        moral_record = dataset.records[1]
        assert 'MoralValue' in moral_record['features']
        assert 'Lily' in moral_record['story']
        assert 'do not steal' in moral_record['story']

    def test_task_generation(self, temp_data_dir):
        """Test generation of different instruction tasks."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        task_types = [task['type'] for task in dataset.tasks]
        
        # Should have all expected task types
        assert 'summarization' in task_types
        assert 'use_words' in task_types  
        assert 'qa' in task_types
        assert 'moral' in task_types  # Because we have MoralValue feature

    def test_chat_formatting(self, temp_data_dir):
        """Test chat format creation."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        task = dataset.tasks[0]
        chat_text = dataset._format_chat(task['prompt'], task['target'])
        
        # Should contain all chat components in correct order
        assert '<|system|>' in chat_text
        assert '<|user|>' in chat_text
        assert '<|assistant|>' in chat_text
        assert '<|endoftext|>' in chat_text
        
        # Verify order
        positions = [
            chat_text.find('<|system|>'),
            chat_text.find('<|user|>'),
            chat_text.find('<|assistant|>'),
            chat_text.find('<|endoftext|>')
        ]
        assert positions == sorted(positions)

    def test_loss_masking(self, temp_data_dir):
        """Test loss masking works correctly."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        task = dataset.tasks[0]
        chat_text = dataset._format_chat(task['prompt'], task['target'])
        input_ids, labels = dataset._tokenize_with_loss_mask(chat_text)
        
        # Count training vs masked tokens
        training_tokens = (labels != -100).sum().item()
        total_tokens = labels.numel()
        
        # Should have some tokens for training, but not all
        assert training_tokens > 0
        assert training_tokens < total_tokens
        
        # Should be reasonable percentage
        training_ratio = training_tokens / total_tokens
        assert 0.02 < training_ratio < 0.5

    def test_dataloader_creation(self, temp_data_dir):
        """Test SFT dataloader creation."""
        dataloader, vocab_size = create_sft_dataloader(
            temp_data_dir,
            batch_size=2,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        assert vocab_size > 0
        assert dataloader.batch_size == 2
        
        # Test batch
        input_ids, labels = next(iter(dataloader))
        assert input_ids.shape == (2, 256)
        assert labels.shape == (2, 256)


class TestSFTTrainingComponents:
    """Test SFT training-specific functionality."""

    def test_loss_computation_with_masking(self):
        """Test that loss computation respects masking."""
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        # Create dummy logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Mask first half of tokens
        labels[:, :seq_len//2] = -100
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_checkpoint_saving_loading(self, tmp_path):
        """Test SFT checkpoint saving and loading."""
        from mini_chat_gpt.sft_train import save_sft_checkpoint, load_pretrained_checkpoint
        
        # Create a small model
        model = create_model(vocab_size=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        save_dir = str(tmp_path / "checkpoints")
        
        # Test saving
        save_sft_checkpoint(model, optimizer, step=100, loss=2.5, save_dir=save_dir)
        
        # Check file was created
        checkpoint_file = os.path.join(save_dir, "sft_checkpoint_step_100.pt")
        assert os.path.exists(checkpoint_file)
        
        # Test loading
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        assert checkpoint['step'] == 100
        assert checkpoint['loss'] == 2.5
        assert checkpoint['model_type'] == 'sft'

    def test_pretrained_checkpoint_loading(self, tmp_path):
        """Test loading pretrained checkpoint into SFT model."""
        from mini_chat_gpt.sft_train import load_pretrained_checkpoint
        
        # Create and save a pretrained model
        model = create_model(vocab_size=1000)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': 1000,
                'dim': 768,
                'n_layers': 12,
                'n_heads': 12,
                'max_seq_len': 1024,
            }
        }
        
        checkpoint_path = str(tmp_path / "pretrained.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Create new model and load checkpoint
        new_model = create_model(vocab_size=1000)
        load_pretrained_checkpoint(checkpoint_path, new_model)
        
        # Models should have same weights
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


class TestSFTGeneration:
    """Test SFT model generation functionality."""

    def test_chat_prompt_formatting(self):
        """Test chat prompt formatting."""
        user_message = "Write a story about a cat."
        system_message = "You are a helpful assistant."
        
        prompt = format_chat_prompt(user_message, system_message)
        
        assert '<|system|>' in prompt
        assert '<|user|>' in prompt
        assert '<|assistant|>' in prompt
        assert user_message in prompt
        assert system_message in prompt

    def test_sft_model_loading(self, tmp_path):
        """Test loading SFT model from checkpoint."""
        # Create a mock SFT checkpoint
        model = create_model(vocab_size=1000)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': 1000,
                'dim': 768,
                'n_layers': 12,
                'n_heads': 12,
                'max_seq_len': 1024,
            },
            'model_type': 'sft'
        }
        
        checkpoint_path = str(tmp_path / "sft_model.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Test loading
        loaded_model, tokenizer = load_sft_model(checkpoint_path, device="cpu")
        
        assert isinstance(loaded_model, GPT)
        assert loaded_model.vocab_size == 1000
        assert isinstance(tokenizer, tiktoken.Encoding)

    @patch('mini_chat_gpt.sft_generate.load_sft_model')
    def test_chat_response_generation(self, mock_load):
        """Test chat response generation."""
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock tokenizer behavior
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "<|system|>System<|user|>User<|assistant|>Response<|endoftext|>"
        
        # Mock model generation
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        # Test generation
        response = generate_chat_response(
            mock_model, mock_tokenizer, 
            "Tell me a story", 
            device="cpu"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestSFTIntegration:
    """Integration tests for SFT pipeline."""

    def test_end_to_end_data_flow(self, tmp_path):
        """Test complete data flow from dataset to model input."""
        # Create sample data
        sample_data = """Features: Dialogue
Words: test, story, integration
Summary: A test story for integration testing.
Story: 

This is a test story. It has dialogue and characters.
"Hello," said the character. "This is a test."
The story teaches us about testing.
<|endoftext|>"""
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(sample_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(sample_data)
        
        # Create dataset and dataloader
        dataloader, vocab_size = create_sft_dataloader(
            str(data_dir),
            batch_size=1,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        # Create model
        model = create_model(vocab_size)
        
        # Test forward pass
        input_ids, labels = next(iter(dataloader))
        
        with torch.no_grad():
            logits, _ = model(input_ids)
            
            # Compute loss with masking
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
        
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_task_type_distribution(self, tmp_path):
        """Test that all task types are generated appropriately."""
        # Create data with all features
        comprehensive_data = """Features: Dialogue
Words: play, park, friends
Summary: Children play together in the park and become friends.
Story: 

Tim and Sam went to the park. They wanted to play together.
"Let's be friends," said Tim. "We can play on the swings."
Sam smiled and nodded. They played happily all day.
<|endoftext|>
Features: Dialogue, BadEnding, MoralValue
Words: share, toys, selfish
Summary: A child learns to share toys after being selfish.
Story: 

Lucy had many toys but didn't want to share with her sister.
"These are mine!" she said angrily. "You can't play with them!"
But when Lucy's toys broke because she played alone, she felt sad.
The moral of the story is: sharing makes everything more fun.
<|endoftext|>"""
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(comprehensive_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(comprehensive_data)
        
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        task_types = [task['type'] for task in dataset.tasks]
        task_counts = {task_type: task_types.count(task_type) for task_type in set(task_types)}
        
        # Should have all task types
        assert 'summarization' in task_counts
        assert 'use_words' in task_counts
        assert 'qa' in task_counts
        assert 'moral' in task_counts  # Because we have MoralValue feature
        
        # Should have reasonable distribution
        assert task_counts['summarization'] == 2  # One per story
        assert task_counts['use_words'] == 2  # One per story
        assert task_counts['moral'] == 1  # Only one story has MoralValue
        assert task_counts['qa'] >= 2  # At least one per story

    def test_special_token_handling(self, tmp_path):
        """Test that special tokens are handled correctly throughout pipeline."""
        # Create data with special characters
        special_data = """Features: Dialogue
Words: special, tokens, test
Summary: Testing special token handling in the pipeline.
Story: 

This story contains special tokens like <|endoftext|> in the text.
"What about other tokens?" asked the character.
The system should handle these gracefully.
<|endoftext|>"""
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(special_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(special_data)
        
        # Should not raise errors with special tokens
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        # Test tokenization
        input_ids, labels = dataset[0]
        assert input_ids.shape == (512,)
        assert labels.shape == (512,)
        
        # Test dataloader
        dataloader, _ = create_sft_dataloader(
            str(data_dir),
            batch_size=1,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        input_ids, labels = next(iter(dataloader))
        assert input_ids.shape == (1, 256)
        assert labels.shape == (1, 256)


class TestSFTErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_dataset_handling(self, tmp_path):
        """Test handling of empty datasets."""
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        # Create empty files
        (data_dir / "TinyStoriesInstruct-train.txt").write_text("")
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text("")
        
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        assert len(dataset.records) == 0
        assert len(dataset.tasks) == 0
        assert len(dataset) == 0

    def test_malformed_data_handling(self, tmp_path):
        """Test handling of malformed data."""
        malformed_data = """Features: Incomplete
Words: missing, story
Summary: This record is missing the story section.
<|endoftext|>
Story: 

This record is missing summary and other fields.
<|endoftext|>"""
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(malformed_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(malformed_data)
        
        # Should handle gracefully
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        # Should filter out incomplete records
        assert len(dataset.records) == 0

    def test_missing_files_error(self, tmp_path):
        """Test error handling for missing files."""
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        # Don't create the required files
        with pytest.raises(FileNotFoundError):
            TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True) 