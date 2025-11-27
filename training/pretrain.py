import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
import sys
from tqdm import tqdm

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger


import json
# from torch.utils.data import IterableDataset


# IterableDataset version (commented out due to distributed training issues)
# class TextDataset(IterableDataset):
#     def __init__(self, file_path, tokenizer, block_size=128):
#         self.file_path = file_path
#         self.tokenizer = tokenizer
#         self.block_size = block_size
#
#     def __iter__(self):
#         buffer = []
#         with open(self.file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 text = line.strip()
#                 if not text:
#                     continue
#
#                 # Try to parse as JSONL
#                 try:
#                     data = json.loads(text)
#                     if isinstance(data, dict) and "text" in data:
#                         text = data["text"]
#                 except json.JSONDecodeError:
#                     pass  # Treat as plain text line
#
#                 tokenized_text = self.tokenizer.encode(text)
#                 buffer.extend(tokenized_text)
#
#                 while len(buffer) >= self.block_size:
#                     yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
#                     buffer = buffer[self.block_size :]


# Regular Dataset version (works better with distributed training)
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        logger = Logger("pretrain")

        # Load and preprocess all data
        logger.info(f"Loading dataset from {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue

                # Try to parse as JSONL
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "text" in data:
                        text = data["text"]
                except json.JSONDecodeError:
                    pass  # Treat as plain text line

                # Tokenize the entire text
                tokenized_text = self.tokenizer.encode(text)
                if len(tokenized_text) > 0:  # Only add non-empty sequences
                    self.examples.append(torch.tensor(tokenized_text, dtype=torch.long))

        logger.info(f"Loaded {len(self.examples)} examples from dataset")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(
    model,
    tokenizer,
    train_file,
    epochs=1,
    batch_size=4,
    lr=1e-4,
    accelerator=None,
    output_dir=None,
    save_steps=1000,
):
    logger = Logger("pretrain")
    logger.info(f"Starting Pretraining on {accelerator.device}...")

    # model.to(device) # Handled by accelerate
    model.train()

    dataset = TextDataset(train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Calculate total steps - now we can use len() since we're using regular Dataset
    # Use ceiling division: (len + batch_size - 1) // batch_size
    total_steps = ((len(dataset) + batch_size - 1) // batch_size) * epochs

    if accelerator.is_main_process:
        logger.info(f"Dataset size: {len(dataset)} examples")
        logger.info(
            f"Total steps: {total_steps} (batch_size={batch_size}, epochs={epochs})"
        )

    # Validate total_steps
    if total_steps <= 0:
        raise ValueError(
            f"Invalid total_steps: {total_steps}. Please check your dataset and ensure it's not empty."
        )

    optimizer = AdamW(model.parameters(), lr=lr)

    # Create learning rate scheduler with warmup
    warmup_steps = max(
        1, int(total_steps * 0.1)
    )  # 10% of total steps for warmup, at least 1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if accelerator.is_main_process:
        logger.info(
            f"Learning rate scheduler: Cosine decay with {warmup_steps} warmup steps ({warmup_steps / total_steps * 100:.1f}% of total steps)"
        )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Setup checkpoint directory
    checkpoint_dir = None
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, "pretrain_latest")
        # Ensure directory exists before any process tries to use it
        if accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        # Wait for main process to create directory before other processes proceed
        accelerator.wait_for_everyone()

    global_step = 0
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(loop):
            # input_ids = batch.to(device) # Handled by accelerate
            input_ids = batch
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # Calculate epoch progress as decimal
            # In distributed training, each process processes a portion of data
            # So we use the local step count, which is correct for progress tracking
            # since total_steps already accounts for the full dataset
            epoch_progress = (
                (global_step / total_steps) * epochs if total_steps > 0 else epoch + 1
            )

            loop.set_postfix(loss=loss.item(), epoch=f"{epoch_progress:.4f}")

            # Log metrics every 10 steps (or adjust as needed)
            if global_step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                accelerator.log(
                    {
                        "train_loss": loss.item(),
                        "epoch": epoch_progress,
                        "step": global_step,
                        "learning_rate": current_lr,
                    }
                )
                logger.log_metrics(
                    {
                        "epoch": f"{epoch_progress:.4f}",
                        "step": global_step,
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )

            # Save checkpoint every save_steps
            if checkpoint_dir and global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(
                        f"Checkpoint saved at step {global_step} to {checkpoint_dir}"
                    )

    logger.info("Pretraining finished.")
    return model


if __name__ == "__main__":
    # Example usage (commented out to avoid import errors if run directly without context)
    pass
