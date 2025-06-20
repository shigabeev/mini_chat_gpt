"""
Comprehensive tests for SFT dataloader functionality.
Tests parsing, task generation, chat formatting, and loss masking.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import tiktoken
from mini_chat_gpt.sft_dataloader import (
    TinyStoriesInstructDataset,
    create_sft_dataloader
)


class TestTinyStoriesInstructDataset:
    """Test the TinyStories-Instruct dataset parser and task generator."""
    
    @pytest.fixture
    def sample_instruct_data(self):
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
Summary: Tom helps a cat stuck in a tree, making both of them happy.
Words: happy, cat, tree
Features: Dialogue
Story: 

Tom was walking in the park when he heard a sad sound. He looked up and saw a cat stuck high in a tree.
"Don't worry, little cat," Tom said. "I will help you."
Tom climbed up carefully and gently picked up the cat. The cat was so happy!
From that day on, the cat and Tom were best friends.
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
    def temp_data_dir(self, tmp_path, sample_instruct_data):
        """Create temporary data directory with sample files."""
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        # Create train file
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(sample_instruct_data)
        # Create validation file
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(sample_instruct_data)
        
        return str(data_dir)
    
    def test_dataset_initialization(self, temp_data_dir):
        """Test basic dataset initialization."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        assert dataset.seq_len == 512
        assert dataset.train == True
        assert len(dataset.records) == 3  # Three stories in sample data
        assert len(dataset.tasks) > 3  # Should generate multiple tasks per story
        assert dataset.tokenizer.n_vocab > 0
    
    def test_record_parsing(self, temp_data_dir):
        """Test parsing of individual records."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        # Check first record
        record = dataset.records[0]
        assert record['features'] == 'Dialogue'
        assert record['words'] == 'quit, oak, gloomy'
        assert 'Sara and Ben' in record['summary']
        assert 'Sara and Ben were playing in the park' in record['story']
        
        # Check record with moral
        moral_record = None
        for record in dataset.records:
            if 'MoralValue' in record.get('features', ''):
                moral_record = record
                break
        
        assert moral_record is not None
        assert 'Lily' in moral_record['story']
        assert 'do not steal' in moral_record['story']
    
    def test_task_generation(self, temp_data_dir):
        """Test generation of different task types."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        task_types = [task['type'] for task in dataset.tasks]
        
        # Should have all task types
        assert 'summarization' in task_types
        assert 'use_words' in task_types  
        assert 'qa' in task_types
        assert 'moral' in task_types  # Because we have MoralValue feature
        
        # Check summarization task
        summarization_task = next(task for task in dataset.tasks if task['type'] == 'summarization')
        assert 'Please provide a brief summary' in summarization_task['prompt']
        assert len(summarization_task['target']) > 0
        
        # Check use_words task
        use_words_task = next(task for task in dataset.tasks if task['type'] == 'use_words')
        assert 'Write a short story that uses all of these words' in use_words_task['prompt']
        assert len(use_words_task['target']) > 0
        
        # Check Q&A task
        qa_task = next(task for task in dataset.tasks if task['type'] == 'qa')
        assert 'Based on this story' in qa_task['prompt']
        assert len(qa_task['target']) > 0
        
        # Check moral task
        moral_task = next(task for task in dataset.tasks if task['type'] == 'moral')
        assert 'What is the moral lesson' in moral_task['prompt']
        assert 'do not steal' in moral_task['target']
    
    def test_chat_formatting(self, temp_data_dir):
        """Test chat format creation."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        task = dataset.tasks[0]
        chat_text = dataset._format_chat(task['prompt'], task['target'])
        
        # Should contain all chat components
        assert '<|system|>' in chat_text
        assert '<|user|>' in chat_text
        assert '<|assistant|>' in chat_text
        assert '<|endoftext|>' in chat_text
        
        # Should be in correct order
        system_pos = chat_text.find('<|system|>')
        user_pos = chat_text.find('<|user|>')
        assistant_pos = chat_text.find('<|assistant|>')
        endoftext_pos = chat_text.find('<|endoftext|>')
        
        assert system_pos < user_pos < assistant_pos < endoftext_pos
    
    def test_tokenization_with_special_tokens(self, temp_data_dir):
        """Test tokenization handles special tokens correctly."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        task = dataset.tasks[0]
        chat_text = dataset._format_chat(task['prompt'], task['target'])
        
        # Should not raise an error with special tokens
        input_ids, labels = dataset._tokenize_with_loss_mask(chat_text)
        
        assert input_ids.shape == (512,)
        assert labels.shape == (512,)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
    
    def test_loss_masking(self, temp_data_dir):
        """Test that loss masking works correctly."""
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
        
        # Typically should be 5-20% of tokens used for training
        training_ratio = training_tokens / total_tokens
        assert 0.02 < training_ratio < 0.5  # Reasonable range
    
    def test_dataset_item_retrieval(self, temp_data_dir):
        """Test __getitem__ method."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=256, train=True)
        
        input_ids, labels = dataset[0]
        
        assert input_ids.shape == (256,)
        assert labels.shape == (256,)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
        
        # Should have some training tokens
        assert (labels != -100).sum().item() > 0
    
    def test_dataset_length(self, temp_data_dir):
        """Test dataset length matches number of tasks."""
        dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        
        assert len(dataset) == len(dataset.tasks)
        assert len(dataset) > 0
    
    def test_different_sequence_lengths(self, temp_data_dir):
        """Test dataset with different sequence lengths."""
        for seq_len in [128, 256, 512, 1024]:
            dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=seq_len, train=True)
            input_ids, labels = dataset[0]
            
            assert input_ids.shape == (seq_len,)
            assert labels.shape == (seq_len,)
    
    def test_train_vs_validation_split(self, temp_data_dir):
        """Test train vs validation data loading."""
        train_dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=True)
        val_dataset = TinyStoriesInstructDataset(temp_data_dir, seq_len=512, train=False)
        
        # Should load same data (since we used same file for both)
        assert len(train_dataset.records) == len(val_dataset.records)
        assert len(train_dataset.tasks) == len(val_dataset.tasks)


class TestSFTDataLoaderCreation:
    """Test SFT dataloader creation and configuration."""
    
    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory with sample files."""
        sample_data = """Features: Dialogue
Words: test, sample, data
Summary: A test story for validation.
Story: 

This is a test story. It has some words and dialogue.
"Hello," said the character. "This is a test."
The character was happy with the test.
<|endoftext|>""" * 10  # Repeat for more data
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(sample_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(sample_data)
        
        return str(data_dir)
    
    def test_create_sft_dataloader_basic(self, temp_data_dir):
        """Test basic SFT dataloader creation."""
        dataloader, vocab_size = create_sft_dataloader(
            temp_data_dir,
            batch_size=4,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        assert vocab_size > 0
        assert dataloader.batch_size == 4
        assert dataloader.drop_last == True
        assert dataloader.pin_memory == True
    
    def test_sft_dataloader_batch_shape(self, temp_data_dir):
        """Test SFT dataloader produces correct batch shapes."""
        dataloader, _ = create_sft_dataloader(
            temp_data_dir,
            batch_size=2,
            seq_len=128,
            train=True,
            num_workers=0
        )
        
        input_ids, labels = next(iter(dataloader))
        
        assert input_ids.shape == (2, 128)
        assert labels.shape == (2, 128)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
    
    def test_sft_dataloader_loss_masking_batch(self, temp_data_dir):
        """Test loss masking works correctly in batches."""
        dataloader, _ = create_sft_dataloader(
            temp_data_dir,
            batch_size=4,
            seq_len=256,
            train=True,
            num_workers=0
        )
        
        input_ids, labels = next(iter(dataloader))
        
        # Each sample should have some training tokens
        for i in range(input_ids.shape[0]):
            training_tokens = (labels[i] != -100).sum().item()
            assert training_tokens > 0
    
    def test_sft_dataloader_consistency(self, temp_data_dir):
        """Test dataloader produces consistent results."""
        # Use validation data for consistency (no shuffle)
        dataloader1, _ = create_sft_dataloader(
            temp_data_dir, batch_size=2, seq_len=128, train=False, num_workers=0
        )
        dataloader2, _ = create_sft_dataloader(
            temp_data_dir, batch_size=2, seq_len=128, train=False, num_workers=0
        )
        
        input_ids1, labels1 = next(iter(dataloader1))
        input_ids2, labels2 = next(iter(dataloader2))
        
        assert torch.equal(input_ids1, input_ids2)
        assert torch.equal(labels1, labels2)


class TestSFTTaskGeneration:
    """Test specific SFT task generation logic."""
    
    def test_summarization_task_creation(self):
        """Test summarization task creation."""
        from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset
        
        dataset = TinyStoriesInstructDataset.__new__(TinyStoriesInstructDataset)
        
        record = {
            'story': 'Once upon a time, there was a cat.',
            'summary': 'A story about a cat.',
            'features': 'None',
            'words': 'cat'
        }
        
        task = dataset._create_summarization_task(record)
        
        assert task['type'] == 'summarization'
        assert 'Please provide a brief summary' in task['prompt']
        assert 'Once upon a time, there was a cat.' in task['prompt']
        assert task['target'] == 'A story about a cat.'
    
    def test_moral_task_creation(self):
        """Test moral extraction task creation."""
        from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset
        
        dataset = TinyStoriesInstructDataset.__new__(TinyStoriesInstructDataset)
        
        record = {
            'story': 'A story about being kind. The moral of the story is: always be kind to others.',
            'summary': 'A moral story.',
            'features': 'MoralValue',
            'words': 'kind'
        }
        
        task = dataset._create_moral_task(record)
        
        assert task is not None
        assert task['type'] == 'moral'
        assert 'What is the moral lesson' in task['prompt']
        assert 'always be kind to others' in task['target']
    
    def test_use_words_task_creation(self):
        """Test use-the-words task creation."""
        from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset
        
        dataset = TinyStoriesInstructDataset.__new__(TinyStoriesInstructDataset)
        
        record = {
            'story': 'The cat climbed the tree and was happy.',
            'summary': 'Cat story.',
            'features': 'None',
            'words': 'cat, tree, happy'
        }
        
        task = dataset._create_use_words_task(record)
        
        assert task['type'] == 'use_words'
        assert 'Write a short story that uses all of these words' in task['prompt']
        assert 'cat, tree, happy' in task['prompt']
        assert task['target'] == 'The cat climbed the tree and was happy.'
    
    def test_qa_task_generation(self):
        """Test Q&A task generation."""
        from mini_chat_gpt.sft_dataloader import TinyStoriesInstructDataset
        
        dataset = TinyStoriesInstructDataset.__new__(TinyStoriesInstructDataset)
        
        record = {
            'story': 'Sara and Ben played in the park. Sara was the main character.',
            'summary': 'Sara and Ben played together.',
            'features': 'Dialogue',
            'words': 'play'
        }
        
        tasks = dataset._create_qa_tasks(record)
        
        assert len(tasks) > 0
        assert all(task['type'] == 'qa' for task in tasks)
        assert any('Based on this story' in task['prompt'] for task in tasks)


class TestSFTSpecialCases:
    """Test edge cases and error conditions."""
    
    def test_empty_data_file(self, tmp_path):
        """Test handling of empty data files."""
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        # Create empty files
        (data_dir / "TinyStoriesInstruct-train.txt").write_text("")
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text("")
        
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        assert len(dataset.records) == 0
        assert len(dataset.tasks) == 0
        assert len(dataset) == 0
    
    def test_malformed_record_handling(self, tmp_path):
        """Test handling of malformed records."""
        malformed_data = """Features: Dialogue
Words: incomplete
Summary: This record is missing the story section.
<|endoftext|>
Story: 

This record is missing summary and features.
<|endoftext|>"""
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(malformed_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(malformed_data)
        
        # Should handle malformed records gracefully
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        # Should filter out malformed records
        assert len(dataset.records) == 0  # Both records are incomplete
    
    def test_very_long_sequences(self, tmp_path):
        """Test handling of very long sequences."""
        long_story = "This is a sentence. " * 200  # Very long story
        
        long_data = f"""Features: Dialogue
Words: long, story, test
Summary: A very long story for testing.
Story: 

{long_story}
<|endoftext|>"""
        
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        (data_dir / "TinyStoriesInstruct-train.txt").write_text(long_data)
        (data_dir / "TinyStoriesInstruct-valid.txt").write_text(long_data)
        
        dataset = TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True)
        
        # Should handle truncation gracefully
        input_ids, labels = dataset[0]
        assert input_ids.shape == (512,)
        assert labels.shape == (512,)
    
    def test_missing_data_files(self, tmp_path):
        """Test error handling for missing data files."""
        data_dir = tmp_path / "TinyStories"
        data_dir.mkdir()
        
        # Don't create the required files
        with pytest.raises(FileNotFoundError):
            TinyStoriesInstructDataset(str(data_dir), seq_len=512, train=True) 