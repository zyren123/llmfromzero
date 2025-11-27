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


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Read file and handle JSONL or plain text
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue

                # Try to parse as JSONL
                try:
                    data = json.loads(text)
                    if isinstance(data, dict) and "text" in data:
                        self.examples.append(data["text"])
                except json.JSONDecodeError:
                    # Treat as plain text line
                    self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        sample = self.examples[i]
        input_ids = self.tokenizer.encode(
            sample,
            add_special_tokens=False,
            max_length=self.block_size,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def train(
    model,
    tokenizer,
    train_file,
    epochs=1,
    batch_size=4,
    lr=1e-4,
    accelerator=None,
):
    logger = Logger("pretrain")
    logger.info(f"Starting Pretraining on {accelerator.device}...")

    # model.to(device) # Handled by accelerate
    model.train()

    dataset = TextDataset(train_file, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=lr)

    # Calculate total steps for scheduler and epoch calculation
    steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size  # Ceiling division
    total_steps = steps_per_epoch * epochs

    # Calculate warmup steps (10% of total steps)
    warmup_steps = int(total_steps * 0.1)

    # Create scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    global_step = 0
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(loop):
            # input_ids = batch.to(device) # Handled by accelerate
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            # labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            global_step += 1
            # Calculate epoch as decimal: current_step / total_steps
            epoch_decimal = global_step / total_steps if total_steps > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0]

            loop.set_postfix(loss=loss.item(), lr=current_lr)

            # Log metrics every 10 steps (or adjust as needed)
            if global_step % 10 == 0:
                accelerator.log(
                    {
                        "train_loss": loss.item(),
                        "epoch": epoch_decimal,
                        "step": global_step,
                        "learning_rate": current_lr,
                    }
                )
                logger.log_metrics(
                    {
                        "epoch": f"{epoch_decimal:.4f}",
                        "step": global_step,
                        "loss": f"{loss.item():.4f}",
                        "learning_rate": f"{current_lr:.6f}",
                    }
                )

    logger.info("Pretraining finished.")
    return model


if __name__ == "__main__":
    # Example usage (commented out to avoid import errors if run directly without context)
    pass
