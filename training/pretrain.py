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


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenized_text = tokenizer.encode(text)

        # Simple sliding window for demonstration
        for i in range(0, len(tokenized_text) - block_size, block_size):
            self.examples.append(tokenized_text[i : i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


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
