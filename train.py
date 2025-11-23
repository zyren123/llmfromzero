import argparse
import torch
import os
import sys
from transformers import PreTrainedTokenizerFast

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.dense import DenseModel, DenseConfig
from models.moe import DeepSeekV3Model, DeepSeekV3Config
from training.pretrain import train as train_pretrain
from training.sft import train_sft
from training.dpo import train_dpo
from utils.logger import Logger


def get_model(model_type, vocab_size, device):
    if model_type == "dense":
        config = DenseConfig(vocab_size=vocab_size)
        model = DenseModel(config)
    elif model_type == "moe":
        config = DeepSeekV3Config(vocab_size=vocab_size)
        model = DeepSeekV3Model(config)
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
        default="dense",
        choices=["dense", "moe"],
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    logger = Logger("train")
    logger.info(f"Starting training with args: {args}")

    # Load Tokenizer
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    logger.info(
        f"Loaded tokenizer from {args.tokenizer_path}, vocab size: {len(tokenizer)}"
    )

    # Initialize Model
    model = get_model(args.model_type, len(tokenizer), args.device)
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
            device=args.device,
        )
    elif args.mode == "sft":
        trained_model = train_sft(
            model=model,
            tokenizer=tokenizer,
            train_file=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    elif args.mode == "dpo":
        trained_model = train_dpo(
            model=model,
            tokenizer=tokenizer,
            train_file=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

    # Save Model
    save_path = os.path.join(args.output_dir, f"{args.model_type}_{args.mode}_final.pt")
    torch.save(trained_model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
