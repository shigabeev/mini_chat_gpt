"""
TinyStories-Instruct dataloader for supervised fine-tuning.
Generates instruction-following tasks from story data.
"""

import os
import re
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import tiktoken


class TinyStoriesInstructDataset(Dataset):
    """TinyStories-Instruct dataset with task generation for SFT."""
    
    def __init__(self, data_path: str, seq_len: int = 1024, train: bool = True):
        self.seq_len = seq_len
        self.train = train
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Special tokens for chat format
        self.system_token = "<|system|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.endoftext_token = "<|endoftext|>"
        
        # Parse the raw instruct data
        self.records = self._parse_instruct_file(data_path, train)
        
        # Generate instruction tasks from records
        self.tasks = self._generate_tasks()
        
        print(f"Loaded {len(self.records)} records, generated {len(self.tasks)} instruction tasks")
    
    def _parse_instruct_file(self, data_path: str, train: bool) -> List[Dict]:
        """Parse TinyStories-Instruct file into structured records."""
        split = "train" if train else "valid"
        filename = f"TinyStoriesInstruct-{split}.txt"
        filepath = os.path.join(data_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"TinyStories-Instruct file not found: {filepath}")
        
        print(f"Parsing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by <|endoftext|> to get individual records
        raw_records = content.split('<|endoftext|>')
        records = []
        
        for raw_record in raw_records:
            raw_record = raw_record.strip()
            if not raw_record:
                continue
                
            record = self._parse_single_record(raw_record)
            if record:
                records.append(record)
        
        return records
    
    def _parse_single_record(self, raw_text: str) -> Optional[Dict]:
        """Parse a single record from raw text."""
        try:
            # Extract fields using regex patterns
            record = {}
            
            # Extract Features
            features_match = re.search(r'Features:\s*(.+)', raw_text)
            record['features'] = features_match.group(1).strip() if features_match else ""
            
            # Extract Words
            words_match = re.search(r'Words:\s*(.+)', raw_text)
            record['words'] = words_match.group(1).strip() if words_match else ""
            
            # Extract Summary
            summary_match = re.search(r'Summary:\s*(.+?)(?=\nStory:|$)', raw_text, re.DOTALL)
            record['summary'] = summary_match.group(1).strip() if summary_match else ""
            
            # Extract Story
            story_match = re.search(r'Story:\s*\n\n(.+)', raw_text, re.DOTALL)
            record['story'] = story_match.group(1).strip() if story_match else ""
            
            # Only return record if it has both story and summary
            if record['story'] and record['summary']:
                return record
            
        except Exception as e:
            print(f"Failed to parse record: {e}")
            
        return None
    
    def _generate_tasks(self) -> List[Dict]:
        """Generate instruction-following tasks from parsed records."""
        tasks = []
        
        for record in self.records:
            # Task A: Summarization
            tasks.append(self._create_summarization_task(record))
            
            # Task B: Moral extraction (if MoralValue feature present)
            if 'MoralValue' in record.get('features', ''):
                moral_task = self._create_moral_task(record)
                if moral_task:
                    tasks.append(moral_task)
            
            # Task C: Use-the-words (creative writing)
            if record.get('words'):
                tasks.append(self._create_use_words_task(record))
            
            # Task D: Q&A generation
            qa_tasks = self._create_qa_tasks(record)
            tasks.extend(qa_tasks)
        
        return tasks
    
    def _create_summarization_task(self, record: Dict) -> Dict:
        """Create a summarization task."""
        prompt = f"Please provide a brief summary of the following story:\n\n{record['story']}"
        target = record['summary']
        
        return {
            'type': 'summarization',
            'prompt': prompt,
            'target': target,
            'record': record
        }
    
    def _create_moral_task(self, record: Dict) -> Optional[Dict]:
        """Create a moral extraction task."""
        story = record['story']
        
        # Look for moral in the story text
        moral_match = re.search(r'The moral of the story is:?\s*(.+?)(?=\n|$)', story, re.DOTALL)
        if not moral_match:
            return None
        
        moral = moral_match.group(1).strip()
        
        prompt = f"What is the moral lesson of this story?\n\n{story}"
        target = moral
        
        return {
            'type': 'moral',
            'prompt': prompt,
            'target': target,
            'record': record
        }
    
    def _create_use_words_task(self, record: Dict) -> Dict:
        """Create a use-the-words creative writing task."""
        words = record['words']
        story = record['story']
        
        prompt = f"Write a short story that uses all of these words: {words}"
        target = story
        
        return {
            'type': 'use_words',
            'prompt': prompt,
            'target': target,
            'record': record
        }
    
    def _create_qa_tasks(self, record: Dict) -> List[Dict]:
        """Create Q&A tasks from the story."""
        story = record['story']
        tasks = []
        
        # Generate simple questions based on story content
        questions = self._generate_questions(story, record)
        
        for question, answer in questions:
            tasks.append({
                'type': 'qa',
                'prompt': f"Based on this story, {question}\n\nStory: {story}",
                'target': answer,
                'record': record
            })
        
        return tasks
    
    def _generate_questions(self, story: str, record: Dict) -> List[Tuple[str, str]]:
        """Generate simple questions and answers from the story."""
        questions = []
        
        # Extract character names (capitalized words that appear multiple times)
        words = re.findall(r'\b[A-Z][a-z]+\b', story)
        characters = [word for word in set(words) if story.count(word) > 1 and word not in ['The', 'A', 'An']]
        
        if characters:
            # Who questions
            main_char = characters[0]
            questions.append((
                f"who is the main character in this story?",
                f"The main character is {main_char}."
            ))
        
        # What happened questions
        if 'summary' in record:
            questions.append((
                f"what happened in this story?",
                record['summary']
            ))
        
        # Features-based questions
        features = record.get('features', '')
        if 'BadEnding' in features:
            questions.append((
                f"does this story have a happy ending?",
                f"No, this story has a bad ending."
            ))
        elif 'Dialogue' in features:
            questions.append((
                f"do the characters talk to each other in this story?",
                f"Yes, the characters have conversations in this story."
            ))
        
        return questions[:2]  # Limit to 2 Q&A pairs per story
    
    def _format_chat(self, prompt: str, target: str) -> str:
        """Format prompt and target into chat format."""
        system_msg = "You are a helpful assistant that answers questions about stories."
        
        chat = f"{self.system_token}{system_msg}{self.user_token}{prompt}{self.assistant_token}{target}{self.endoftext_token}"
        return chat
    
    def _tokenize_with_loss_mask(self, chat_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize chat and create loss mask (only train on assistant tokens)."""
        # Find where assistant response starts
        assistant_start = chat_text.find(self.assistant_token)
        if assistant_start == -1:
            raise ValueError("Assistant token not found in chat text")
        
        assistant_start += len(self.assistant_token)
        
        # Tokenize full text (allow special tokens)
        tokens = self.tokenizer.encode(chat_text, allowed_special={"<|endoftext|>"})
        
        # Create loss mask: -100 for tokens we don't want to train on
        labels = tokens.copy()
        
        # Find assistant start in token space
        prefix_text = chat_text[:assistant_start]
        prefix_tokens = self.tokenizer.encode(prefix_text, allowed_special={"<|endoftext|>"})
        
        # Mask everything before assistant response
        for i in range(len(prefix_tokens)):
            if i < len(labels):
                labels[i] = -100
        
        # Truncate to max sequence length
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
            labels = labels[:self.seq_len]
        
        # Pad to sequence length
        while len(tokens) < self.seq_len:
            tokens.append(self.tokenizer.eot_token)  # Use end-of-text token for padding
            labels.append(-100)  # Don't compute loss on padding
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        task = self.tasks[idx]
        
        # Format as chat
        chat_text = self._format_chat(task['prompt'], task['target'])
        
        # Tokenize with loss masking
        input_ids, labels = self._tokenize_with_loss_mask(chat_text)
        
        return input_ids, labels


def create_sft_dataloader(
    data_path: str,
    batch_size: int,
    seq_len: int = 1024,
    train: bool = True,
    num_workers: int = 4,
) -> Tuple[DataLoader, int]:
    """Create SFT dataloader for TinyStories-Instruct."""
    
    dataset = TinyStoriesInstructDataset(data_path, seq_len, train)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader, dataset.tokenizer.n_vocab


if __name__ == "__main__":
    # Test the SFT dataloader
    print("Testing TinyStories-Instruct SFT dataloader...")
    print("=" * 60)
    
    # Test with small sample
    try:
        train_loader, vocab_size = create_sft_dataloader(
            "./TinyStories", 
            batch_size=2, 
            seq_len=512, 
            train=True,
            num_workers=0
        )
        
        print(f"Vocab size: {vocab_size}")
        print(f"Dataset length: {len(train_loader.dataset)}")
        
        # Test a few batches
        tokenizer = tiktoken.get_encoding("gpt2")
        
        for i, (input_ids, labels) in enumerate(train_loader):
            if i >= 2:  # Only test first 2 batches
                break
                
            print(f"\nBatch {i+1}:")
            print(f"Input shape: {input_ids.shape}")
            print(f"Labels shape: {labels.shape}")
            
            # Decode first sample to verify format
            sample_input = input_ids[0]
            sample_labels = labels[0]
            
            # Find non-padded tokens
            non_pad_mask = sample_input != tokenizer.eot_token
            actual_length = non_pad_mask.sum().item()
            
            print(f"Actual sequence length: {actual_length}")
            
            # Decode the text
            text = tokenizer.decode(sample_input[:actual_length].tolist())
            print(f"Sample text: {text[:200]}...")
            
            # Show loss mask
            train_tokens = (sample_labels[:actual_length] != -100).sum().item()
            print(f"Tokens to train on: {train_tokens}/{actual_length}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure TinyStories-Instruct files are available in ./TinyStories/") 