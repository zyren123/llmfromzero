import os
import json
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from utils.logger import Logger
import argparse


def batch_iterator(files, logger=None):
    """
    Yields text data from files (supports .txt and .jsonl).
    """
    for file_path in files:
        if not os.path.exists(file_path):
            if logger:
                logger.warning(f"File {file_path} not found.")
            else:
                print(f"Warning: File {file_path} not found.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            yield data["text"]
                    except json.JSONDecodeError:
                        if logger:
                            logger.warning(f"Failed to decode JSON line in {file_path}")
                        else:
                            print(f"Warning: Failed to decode JSON line in {file_path}")
            else:
                # Assume plain text
                for line in f:
                    yield line


def train_tokenizer(files, vocab_size=32000, save_path="mini_tokenizer"):
    """
    Trains a BPE tokenizer from scratch.
    """
    logger = Logger("train_tokenizer")
    logger.info(f"Initializing tokenizer training with vocab_size={vocab_size}")

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-normalization and pre-tokenization
    # We use ByteLevel which is common for GPT-2/Llama style models
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<think>",
            "</think>",
        ],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    # Train
    logger.info(f"Training tokenizer on {files}...")
    # tokenizer.train(files, trainer)
    tokenizer.train_from_iterator(batch_iterator(files, logger), trainer=trainer)

    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2
    assert tokenizer.token_to_id("<think>") == 3
    assert tokenizer.token_to_id("</think>") == 4

    # Post-processor (optional, but good for consistency)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save the raw tokenizer
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))

    # Wrap in Transformers PreTrainedTokenizerFast for easy usage
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",  # BPE usually doesn't have UNK if byte fallback is on, but setting safe default
        pad_token="<|endoftext|>",  # Using EOS as pad is common in simple setups
        additional_special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
    )

    fast_tokenizer.save_pretrained(save_path)
    logger.info(f"Tokenizer saved to {save_path}")
    return fast_tokenizer


if __name__ == "__main__":
    # Example usage with the actual data file
    parser = argparse.ArgumentParser(description="Tokenizer Training Script")
    parser.add_argument("--data_path", type=str, default="data/pretrain_hq.jsonl")
    args = parser.parse_args()
    data_file = args.data_path

    # Check if the data file exists, otherwise create a dummy one for testing
    if not os.path.exists(data_file):
        print(
            f"Data file {data_file} not found. Creating a dummy jsonl for demonstration."
        )
        dummy_file = "dummy_data.jsonl"
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "text": "Hello world! This is a test dataset for training a tokenizer."
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps({"text": "Deep learning is amazing. Transformers are cool."})
                + "\n"
            )
        files_to_train = [dummy_file]
    else:
        files_to_train = [data_file]

    train_tokenizer(files_to_train, vocab_size=6400)
