# Educational LLM Project Summary

This project implements a complete pipeline for training Large Language Models (LLMs) from scratch, designed for educational purposes. It covers tokenizer training, multiple model architectures, and various training stages.

## 1. Project Structure

```
llm-from-scratch/
├── tokenizer/
│   └── train_tokenizer.py  # BPE Tokenizer training script
├── models/
│   ├── dense.py            # Qwen2.5-style Dense Model
│   ├── moe.py              # DeepSeek-V3-style MoE Model (MLA + DeepSeekMoE)
│   └── vlm.py              # Vision-Language Model (ViT + Projector + LLM)
├── training/
│   ├── pretrain.py         # Causal Language Modeling Loop
│   ├── sft.py              # Supervised Fine-Tuning Loop (Instruction Tuning)
│   └── dpo.py              # Direct Preference Optimization Loop
└── verify_project.py       # End-to-end verification script
```

## 2. Key Features Implemented

### Tokenizer
- **BPE Training**: Uses `tokenizers` library to train a Byte-Pair Encoding tokenizer from raw text.
- **Transformers Compatibility**: Wraps the trained tokenizer in `PreTrainedTokenizerFast` for seamless integration with Hugging Face `transformers`.

### Model Architectures
- **Dense (Qwen2.5 Style)**:
  - Implements RMSNorm, SwiGLU, and RoPE.
  - Supports Grouped Query Attention (GQA).
- **MoE (DeepSeek-V3 Style)**:
  - **MLA (Multi-Head Latent Attention)**: Implements Low-Rank Key-Value compression.
  - **DeepSeekMoE**: Implements Fine-Grained Expert Segmentation with Shared Experts and Routed Experts.
- **VLM**:
  - Combines a simple Vision Encoder (ViT-style) with the Dense LLM.
  - Uses an MLP Projector to map image features to the text embedding space.

### Training Loops
- **Pretrain**: Standard next-token prediction on raw text files.
- **SFT**: Instruction tuning with masking of user prompts.
- **DPO**: Preference alignment using chosen/rejected pairs with manual DPO loss implementation.

### Think Mode (Reasoning)
- **Special Tokens**: Tokenizer supports `<think>` and `</think>` tags.
- **Inference**: `inference.py` demonstrates how to parse and display the "Thinking Process" separately from the final answer, similar to reasoning models.

## 3. Verification

A `verify_project.py` script is provided to test the entire pipeline. It:
1. Generates dummy data for all stages.
2. Trains a tokenizer.
3. Instantiates and runs a training step for all 3 models and 3 training modes.

To run the verification:
```bash
python verify_project.py
```
