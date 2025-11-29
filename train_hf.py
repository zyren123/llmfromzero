import argparse
import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from training.pretrain import TextDataset
from models.dense import LuluModel, LuluConfig
from models.moe import LuluMoeModel, LuluMoeConfig
from models.vlm import LuluVLModel, LuluVLConfig
from models.minimind import MiniMindModel, MiniMindConfig, MiniMindForCausalLM

# Add current directory to path to ensure local modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_model(model_type, vocab_size, model_path=None, **kwargs):
    """Initialize or load a model.

    Args:
        model_type: Type of model architecture (lulu, lulu_moe, lulu_vl, minimind)
        vocab_size: Vocabulary size for new models
        model_path: Optional path to load pretrained model from

    Returns:
        Model instance
    """
    if model_path and os.path.exists(model_path):
        # Load from pretrained checkpoint
        if model_type == "lulu":
            model = LuluModel.from_pretrained(model_path)
        elif model_type == "lulu_moe":
            model = LuluMoeModel.from_pretrained(model_path)
        elif model_type == "lulu_vl":
            model = LuluVLModel.from_pretrained(model_path)
        elif model_type == "minimind":
            model = MiniMindForCausalLM.from_pretrained(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        # Initialize from scratch
        if model_type == "lulu":
            config = LuluConfig(vocab_size=vocab_size, **kwargs)
            model = LuluModel(config)
        elif model_type == "lulu_moe":
            config = LuluMoeConfig(vocab_size=vocab_size, **kwargs)
            model = LuluMoeModel(config)
        elif model_type == "lulu_vl":
            config = LuluVLConfig(vocab_size=vocab_size, **kwargs)
            model = LuluVLModel(config)
        elif model_type == "minimind":
            config = MiniMindConfig(vocab_size=vocab_size)
            model = MiniMindForCausalLM(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Trainer Script")
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "sft", "dpo"],
        help="Training mode",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lulu",
        choices=["lulu", "lulu_moe", "lulu_vl", "minimind"],
        help="Model architecture",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="mini_tokenizer",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints_hf",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size per device"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lulu-training-hf",
        help="WandB project name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--use_gated_attention",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Use gated attention",
    )
    parser.add_argument(
        "--block_size", type=int, default=512, help="Block size for tokenizer"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training")

    args = parser.parse_args()
    args.use_gated_attention = args.use_gated_attention == "True"

    # Load Tokenizer
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print(
        f"Tokenizer loaded. Pad Token ID: {tokenizer.pad_token_id}, EOS Token ID: {tokenizer.eos_token_id}"
    )

    # Initialize Model
    model = get_model(
        args.model_type,
        len(tokenizer),
        args.model_path,
        use_gated_attention=args.use_gated_attention,
    )

    print(f"Model {args.model_type} initialized.")

    # Print trainable parameters
    trainable_params = model.num_parameters(only_trainable=True)
    all_params = model.num_parameters(only_trainable=False)
    print(
        f"trainable params: {trainable_params/1e6:.2f}M || all params: {all_params/1e6:.2f}M || trainable%: {100 * trainable_params / all_params:.2f}"
    )

    # Prepare Dataset
    dataset = TextDataset(args.data_path, tokenizer, block_size=args.block_size)
    print(f"Dataset loaded with {len(dataset)} examples.")

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if args.wandb_project else "none",
        run_name=f"{args.model_type}-pretrain",
        remove_unused_columns=False,
        save_safetensors=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
