import argparse
import torch
import os
import sys
from transformers import PreTrainedTokenizerFast, AutoTokenizer
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


def get_model(model_type, vocab_size, model_path=None, **kwargs):
    """Initialize or load a model.

    Args:
        model_type: Type of model architecture (lulu, lulu_moe, lulu_vl)
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lulu-training",
        help="WandB project name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (optional, will initialize from scratch if not provided)",
    )

    parser.add_argument(
        "--use_gated_attention",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Use gated attention",
    )

    args = parser.parse_args()
    args.use_gated_attention = args.use_gated_attention == "True"
    # Initialize Accelerator with gradient accumulation
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    accelerator.init_trackers(project_name=args.wandb_project, config=vars(args))

    logger = Logger("train", is_main_process=accelerator.is_main_process)

    logger.info(f"Starting training with args: {args}")
    logger.info(f"Accelerator device: {accelerator.device}")

    # Load Tokenizer
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = "<|endoftext|>"
    logger.info(
        f"Tokenizer loaded. Pad Token ID: {tokenizer.pad_token_id}, EOS Token ID: {tokenizer.eos_token_id}"
    )
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(
        f"Loaded tokenizer from {args.tokenizer_path}, vocab size: {len(tokenizer)}"
    )

    # Initialize or Load Model
    model = get_model(
        args.model_type,
        len(tokenizer),
        args.model_path,
        use_gated_attention=args.use_gated_attention,
    )
    if args.model_path:
        logger.info(f"Loaded {args.model_type} model from {args.model_path}")
    else:
        logger.info(f"Initialized {args.model_type} model from scratch")

    # Create output directory (only on main process)
    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    accelerator.wait_for_everyone()  # Wait for main process to create directory

    # Dispatch to training function
    if args.mode == "pretrain":
        trained_model = train_pretrain(
            model=model,
            tokenizer=tokenizer,
            train_file=args.data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    # Save Model (only on main process)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(trained_model)
        # Convert model to fp16 before saving to reduce model size
        unwrapped_model = unwrapped_model.to(torch.float16)
        unwrapped_model.save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Model and tokenizer saved to {args.output_dir} (fp16)")

    accelerator.end_training()


if __name__ == "__main__":
    main()
