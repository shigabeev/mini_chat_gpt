# TinyStories Supervised Fine-Tuning (SFT)

Complete implementation of supervised fine-tuning for your pretrained TinyStories GPT model to create an instruction-following chatbot.

## 🎯 Overview

This SFT implementation transforms your pretrained TinyStories model into an instruction-following assistant that can:

- **Summarize stories** - Provide concise summaries of given stories
- **Extract morals** - Identify moral lessons from stories with MoralValue features  
- **Creative writing** - Generate stories using specific words
- **Answer questions** - Respond to who/what/why questions about stories

## 📁 File Structure

```
mini_chat_gpt/
├── sft_dataloader.py     # TinyStories-Instruct dataset parser & task generator
├── sft_train.py          # SFT training script with loss masking
├── sft_generate.py       # Chat-format generation for SFT models
└── model.py              # Shared GPT model (unchanged)

# Config files
├── sft_config.yaml       # SFT training configuration
└── prepare_sft_data.py   # Data preparation script

# Data (after preparation)
TinyStories/
├── TinyStoriesInstruct-train.txt  # Training data
└── TinyStoriesInstruct-valid.txt  # Validation data
```

## 🚀 Quick Start

### 1. Data Preparation

```bash
# Prepare TinyStories-Instruct dataset
python prepare_sft_data.py
```

This creates sample data for testing. Replace with real TinyStoriesInstruct files when available.

### 2. Update Configuration

Edit `sft_config.yaml` to point to your pretrained checkpoint:

```yaml
sft:
  pretrained_checkpoint: "/path/to/your/pretrained/checkpoint.pt"
  # ... other settings
```

### 3. Start SFT Training

```bash
# Single GPU training
python mini_chat_gpt/sft_train.py

# Multi-GPU training (if needed)
torchrun --nproc_per_node=2 mini_chat_gpt/sft_train.py
```

### 4. Test Your SFT Model

```bash
# Run capability tests
python mini_chat_gpt/sft_generate.py \
  --checkpoint /path/to/sft_checkpoint.pt \
  --test

# Interactive chat
python mini_chat_gpt/sft_generate.py \
  --checkpoint /path/to/sft_checkpoint.pt \
  --interactive

# Single prompt test
python mini_chat_gpt/sft_generate.py \
  --checkpoint /path/to/sft_checkpoint.pt \
  --prompt "Write a story using these words: cat, tree, happy"
```

## 📊 Task Generation

The SFT dataloader automatically generates multiple instruction tasks from each story:

### Task Types

1. **Summarization** (1 per story)
   - **Prompt**: "Please provide a brief summary of the following story: [STORY]"
   - **Target**: Story summary

2. **Moral Extraction** (if MoralValue feature present)
   - **Prompt**: "What is the moral lesson of this story? [STORY]"
   - **Target**: Extracted moral sentence

3. **Use-the-Words** (1 per story)
   - **Prompt**: "Write a short story that uses all of these words: [WORDS]"
   - **Target**: Original story

4. **Q&A Generation** (1-2 per story)
   - Character questions: "Who is the main character?"
   - Plot questions: "What happened in this story?"
   - Feature questions: "Does this story have a happy ending?"

### Chat Format

All tasks are wrapped in chat format:
```
<|system|>You are a helpful assistant that answers questions about stories.<|user|>[PROMPT]<|assistant|>[TARGET]<|endoftext|>
```

## 🔧 Training Details

### Loss Masking

Only assistant tokens are used for loss computation:
- System and user tokens are masked with `-100`
- Only the assistant's response contributes to training loss
- Typically ~5-10% of tokens are used for training

### Hyperparameters

**Key differences from pretraining:**
- **Learning Rate**: `2e-5` (1/10th of pretraining)
- **Steps**: `10,000` (much fewer than pretraining)
- **Batch Size**: `4` (smaller for longer sequences)
- **Weight Decay**: `0.01` (lower for fine-tuning)

### Expected Training Time

- **Single GPU**: ~2-3 hours for 10K steps
- **Memory**: ~6-8GB VRAM (batch_size=4, seq_len=1024)

## 📈 Monitoring Training

### WandB Metrics

- `sft/train_loss` - Training loss (should decrease to ~2.0-3.0)
- `sft/val_loss` - Validation loss (monitor for overfitting)
- `sft/lr` - Learning rate schedule
- `sft/tokens_per_second` - Training throughput

### Expected Loss Curves

- **Initial loss**: ~6-8 (when starting from pretrained)
- **Target loss**: ~2.0-3.0 (good instruction following)
- **Overfitting**: Validation loss starts increasing

## 🎭 Example Interactions

### Story Summarization
```
User: Please provide a brief summary of the following story: [long story]
Assistant: Sara and Ben were playing in the park, but Sara wanted to go home because it was cold and dark. Ben convinced her to stay and play, but eventually agreed to go home and have hot cocoa.
```

### Creative Writing
```
User: Write a short story that uses all of these words: cat, tree, happy
Assistant: Tom was walking in the park when he heard a sad sound. He looked up and saw a cat stuck high in a tree. Tom climbed up and rescued the cat. Both Tom and the cat were very happy, and they became best friends.
```

### Q&A
```
User: Based on this story, who is the main character? [story about Sara]
Assistant: The main character is Sara.
```

## 🔍 Evaluation

### Automatic Metrics

After training, evaluate your model with:
```python
from mini_chat_gpt.sft_generate import test_sft_capabilities

# Test different capabilities
test_sft_capabilities(model, tokenizer)
```

### Manual Evaluation

Test instruction following quality:
1. **Summarization accuracy** - Are summaries correct and concise?
2. **Story generation** - Does it use all specified words?
3. **Q&A correctness** - Are answers factually correct?
4. **Chat format** - Does it follow conversation structure?

## ⚠️ Common Issues

### 1. Loss Not Decreasing
- **Check pretrained checkpoint** - Ensure it loads correctly
- **Verify data format** - Test with `prepare_sft_data.py`
- **Learning rate** - Try 1e-5 to 5e-5 range

### 2. Memory Issues
- **Reduce batch_size** - Try 2 or 1
- **Reduce seq_len** - Try 512 instead of 1024
- **Disable torch.compile** - Set `compile: false`

### 3. Poor Generation Quality
- **More training steps** - Try 15K-20K steps
- **Check validation loss** - Ensure not overfitting
- **Temperature tuning** - Try 0.3-1.0 range

### 4. Tokenizer Errors
- **Special tokens** - Ensure `allowed_special` is set correctly
- **Encoding issues** - Check UTF-8 encoding in data files

## 🔄 Next Steps After SFT

Once SFT is working well:

1. **RLHF (Reinforcement Learning from Human Feedback)**
   - Train reward model on human preferences
   - Use PPO to align with human values

2. **Constitutional AI**
   - Self-critique and improvement
   - Safety filtering

3. **Domain Specialization**
   - Fine-tune on specific domains (education, entertainment)
   - Add task-specific capabilities

## 📚 Configuration Reference

### sft_config.yaml Structure

```yaml
sft:
  pretrained_checkpoint: "/path/to/pretrained.pt"
  dataset_path: "./TinyStories"
  learning_rate: 2e-5
  max_steps: 10000
  warmup_steps: 500
  eval_interval: 500
  save_interval: 1000
  save_dir: "/path/to/sft/checkpoints"
  resume_from: null

training:
  batch_size: 4
  weight_decay: 0.01
  # ... other params

system:
  multi_gpu: false  # Start with single GPU
  compile: true
  mixed_precision: true
```

## 🤝 Contributing

To extend this SFT implementation:

1. **Add new task types** - Modify `_generate_tasks()` in `sft_dataloader.py`
2. **Improve parsing** - Enhance `_parse_single_record()` for better field extraction
3. **Add evaluation metrics** - Extend `sft_generate.py` with automated evaluation
4. **Multi-turn conversations** - Support conversation history

## 📄 License

Same as the main project. Use responsibly and follow the peaceful psychopath code style! 😄 