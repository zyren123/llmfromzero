import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator
import os
import sys
from tqdm import tqdm

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger


import json
from torch.utils.data import IterableDataset


class TextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue

                # Try to parse as JSONL
                try:
                    data = json.loads(text)
                    if isinstance(data, dict) and "text" in data:
                        text = data["text"]
                except json.JSONDecodeError:
                    pass  # Treat as plain text line

                tokenized_text = self.tokenizer.encode(text)
                buffer.extend(tokenized_text)

                while len(buffer) >= self.block_size:
                    yield torch.tensor(buffer[: self.block_size], dtype=torch.long)
                    buffer = buffer[self.block_size :]


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

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

            loop.set_postfix(loss=loss.item())

            # Log metrics every 10 steps (or adjust as needed)
            if i % 10 == 0:
                accelerator.log(
                    {"train_loss": loss.item(), "epoch": epoch + 1, "step": i}
                )
                logger.log_metrics(
                    {"epoch": epoch + 1, "step": i, "loss": f"{loss.item():.4f}"}
                )

    logger.info("Pretraining finished.")
    return model


if __name__ == "__main__":
    # Example usage (commented out to avoid import errors if run directly without context)
    pass
