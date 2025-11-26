import argparse
import torch
import os
import sys
from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator
import wandb

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dense import LuluModel, LuluConfig
from models.moe import LuluMoeModel, LuluMoeConfig
from models.vlm import LuluVLModel, LuluVLConfig
from training.pretrain import train as train_pretrain
from training.sft import train_sft
from training.dpo import train_dpo
from utils.logger import Logger


def get_model(model_type, vocab_size):
    if model_type == "lulu":
        config = LuluConfig(vocab_size=vocab_size)
        model = LuluModel(config)
    elif model_type == "lulu_moe":
        config = LuluMoeConfig(vocab_size=vocab_size)
        model = LuluMoeModel(config)
    elif model_type == "lulu_vl":
        config = LuluVLConfig(vocab_size=vocab_size)
        model = LuluVLModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "sft", "dpo"],
        help="Training mode",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lulu",
        choices=["lulu", "lulu_moe", "lulu_vl"],
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
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lulu-training",
        help="WandB project name",
    )

    args = parser.parse_args()

    logger = Logger("train")

    # Initialize Accelerator
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(project_name=args.wandb_project, config=vars(args))

    logger.info(f"Starting training with args: {args}")
    logger.info(f"Accelerator device: {accelerator.device}")

    # Load Tokenizer
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    logger.info(
        f"Loaded tokenizer from {args.tokenizer_path}, vocab size: {len(tokenizer)}"
    )

    # Initialize Model
    model = get_model(args.model_type, len(tokenizer))
    logger.info(f"Initialized {args.model_type} model")

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Dispatch to training function
    if args.mode == "pretrain":
        trained_model = train_pretrain(
            model=model,
            tokenizer=tokenizer,
            train_file=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            accelerator=accelerator,
        )
    elif args.mode == "sft":
        trained_model = train_sft(
            model=model,
            tokenizer=tokenizer,
            train_file=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            accelerator=accelerator,
        )
    elif args.mode == "dpo":
        trained_model = train_dpo(
            model=model,
            tokenizer=tokenizer,
            train_file=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            accelerator=accelerator,
        )

    # Save Model
    # Save Model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(trained_model)
    unwrapped_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
