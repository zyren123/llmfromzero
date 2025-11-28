import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import json
import os
import sys
from training.trainer import train_model

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-calculate tokens for matching assistant header
        # We want to match: <|im_start|>assistant\n
        # This sequence marks the beginning of an assistant message
        # Note: We use encode to ensure we get the exact token IDs used by the model
        self.assistant_header_ids = tokenizer.encode(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        )
        self.eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # Expects a list of dicts with "conversations" key
        # Format: [{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue

                # Try to parse as JSONL
                try:
                    data = json.loads(text)
                    if isinstance(data, dict) and "conversations" in data:
                        self.examples.append(data["conversations"])
                except json.JSONDecodeError:
                    # Treat as plain text line
                    self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        conversations = self.examples[i]

        # Apply chat template to get the full text
        full_text = self.tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Initialize labels as -100 (masked)
        labels = [-100] * len(input_ids)

        # Search for assistant headers and unmask the following content
        i = 0
        header_len = len(self.assistant_header_ids)

        while i < len(input_ids):
            # Check if we found the assistant header
            # We look for the sequence <|im_start|>assistant\n
            if input_ids[i : i + header_len] == self.assistant_header_ids:
                # Found header
                # The actual content starts after the header
                start_content = i + header_len

                # Find the end of the message (EOS token)
                end_content = start_content
                while end_content < len(input_ids):
                    if input_ids[end_content] == self.eos_id:
                        break
                    end_content += 1

                # If we found an EOS (or reached end of sequence), set labels
                # We want to train on: content + EOS
                # So range is [start_content, end_content + 1] (if EOS exists)

                if end_content < len(input_ids):
                    # Found EOS, include it in loss
                    # range is exclusive at the end, so +1 to include EOS
                    labels[start_content : end_content + 1] = input_ids[
                        start_content : end_content + 1
                    ]

                    # Move i to after this block
                    i = end_content + 1
                else:
                    # Did not find EOS (truncated?), still train on content?
                    # Let's include content up to end.
                    labels[start_content:] = input_ids[start_content:]
                    i = len(input_ids)
            else:
                i += 1

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
    gradient_accumulation_steps=1,
    accelerator=None,
):
    dataset = SFTDataset(train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    # Calculate total steps for scheduler
    steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    optimizer_steps_per_epoch = (
        steps_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    # Calculate warmup steps (10% of total optimizer steps)
    warmup_steps = int(total_optimizer_steps * 0.1)

    # Create scheduler with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    return train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        accelerator=accelerator,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        scheduler=scheduler,
        logger_name="sft",
    )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./lulu_tokenizer")
    dataset = SFTDataset("data/sft_mini_512.jsonl", tokenizer)
    x = dataset[12]
    print(x)
