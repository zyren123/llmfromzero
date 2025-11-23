import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
import json
from tqdm import tqdm
import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expects a list of dicts: [{"instruction": "...", "output": "..."}]
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        instruction = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{item['output']}<|im_end|>"

        # Tokenize
        instruction_ids = self.tokenizer.encode(instruction, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        input_ids = instruction_ids + response_ids
        labels = [-100] * len(instruction_ids) + response_ids  # Mask instruction

        # Truncate or Pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_sft(
    model,
    tokenizer,
    train_file,
    epochs=1,
    batch_size=2,
    lr=5e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    logger = Logger("sft")
    logger.info(f"Starting SFT on {device}...")
    model.to(device)
    model.train()

    dataset = SFTDataset(train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"SFT Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

            # Log metrics every 10 steps
            if i % 10 == 0:
                logger.log_metrics(
                    {"epoch": epoch + 1, "step": i, "loss": f"{loss.item():.4f}"}
                )

    logger.info("SFT finished.")
    return model
