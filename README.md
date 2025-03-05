# DeepSeek-R1 Fine-Tune
# Mental Health AI Chatbot - Fine-Tuning DeepSeek-R1-Distill-Llama-8B with Unsloth

## Overview
This project fine-tunes a large language model on mental health counseling conversations to improve the quality of AI-driven counseling support. It leverages **Unsloth for efficient fine-tuning**, **LoRA adapters**, and **Hugging Face datasets**.

## Technologies Used
- **Dataset**: [Sulav/Mental_Health_Counseling_Conversations_ShareGPT](https://huggingface.co/datasets/Sulav/mental_health_counseling_conversations_sharegpt)
- **Model**: [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B)
- **Libraries**: PyTorch, Hugging Face Transformers, Unsloth, Pandas, NumPy

## Methodology
1. **Dataset Preprocessing**
   - Converted ShareGPT-style JSON (`"from", "value"`) into standard `"role", "content"` format.
   - Applied `get_chat_template()` to standardize chat interactions.

2. **Fine-Tuning**
   - Enabled **4-bit quantization** for lower memory usage.
   - Applied **LoRA adapters** (`r=16`, `lora_alpha=16`) for parameter-efficient training.
   - Used **gradient checkpointing** for further memory optimization.

3. **Inference**
   - Enabled **2x faster inference** using Unsloth patches.
   - Used **custom chat formatting** to structure responses.

## Results
- Model fine-tuning improved response quality and **context awareness**.
- The chatbot provided empathetic and structured responses suitable for mental health support.

## Future Work
- Integrate **RLHF** to align responses with professional counseling best practices.
- Deploy as an API or chatbot for real-world use.

## How to Use
```python
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
import torch

# Load the fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    load_in_4bit=True
)

# Apply the chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

# Inference example
messages = [{"role": "user", "content": "I am mentally very upset because I failed my job interview today"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)


This project fine-tunes the **DeepSeek-R1-Distill-Llama-8B** model using the **Unsloth** library. The model is optimized with **LoRA (Low-Rank Adaptation)** for efficient training, leveraging 4-bit quantization to balance performance and memory usage. 

### LLM Model Configuration

#### Model Used
- **LLM (Large Language Model):** `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Library:** Unsloth
- **Architecture:** LLaMA-based
- **Quantization:** 4-bit

### Important Hyperparameters
##### Sequence Length
- `max_seq_length = 2048`

#### LoRA (Low-Rank Adaptation) Parameters
- `r = 16` (LoRA rank)
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (Layers modified)
- `lora_alpha = 16` (Scaling factor for LoRA)
- `lora_dropout = 0`
- `use_gradient_checkpointing = "unsloth"`
- `random_state = 3407`
- `use_rslora = False`

#### Model Inference Parameters
- `temperature = 0.6` (Controls randomness)
- `min_p = 0.1` (Minimum probability threshold for token selection)
- `max_new_tokens = 64` (Limits the response length)
- `use_cache = True` (Enables faster token generation)

#### Tokenizer
- **Chat template:** `"llama-3.2"`
- `tokenizer.pad_token = tokenizer.eos_token`

## Training Dataset
- **Dataset:** `"Sulav/mental_health_counseling_conversations_sharegpt"`
- **Preprocessing:** `standardize_sharegpt()` (Converts ShareGPT format to Hugging Face format)

