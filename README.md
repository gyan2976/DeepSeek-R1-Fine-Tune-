# DeepSeek-R1-Fine-Tune
This project fine-tunes the DeepSeek-R1-Distill-Llama-8B model using the Unsloth library. The model is optimized with LoRA (Low-Rank Adaptation) for efficient training, leveraging 4-bit quantization to balance performance and memory usage. 

## LLM Model Configuration

## Model Used
- **LLM (Large Language Model):** `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Library:** Unsloth
- **Architecture:** LLaMA-based
- **Quantization:** 4-bit

## Important Hyperparameters
### Sequence Length
- `max_seq_length = 2048`

### LoRA (Low-Rank Adaptation) Parameters
- `r = 16` (LoRA rank)
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (Layers modified)
- `lora_alpha = 16` (Scaling factor for LoRA)
- `lora_dropout = 0`
- `use_gradient_checkpointing = "unsloth"`
- `random_state = 3407`
- `use_rslora = False`

## Model Inference Parameters
- `temperature = 0.6` (Controls randomness)
- `min_p = 0.1` (Minimum probability threshold for token selection)
- `max_new_tokens = 64` (Limits the response length)
- `use_cache = True` (Enables faster token generation)

## Tokenizer
- **Chat template:** `"llama-3.2"`
- `tokenizer.pad_token = tokenizer.eos_token`

## Training Dataset
- **Dataset:** `"Sulav/mental_health_counseling_conversations_sharegpt"`
- **Preprocessing:** `standardize_sharegpt()` (Converts ShareGPT format to Hugging Face format)

