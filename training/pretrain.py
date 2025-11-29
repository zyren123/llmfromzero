from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from training.trainer import train_model
import json


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size

        # Read file and handle JSONL or plain text
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            text = line.strip()
            if not text:
                continue

            try:
                data = json.loads(text)
                if isinstance(data, dict) and "text" in data:
                    text_content = data["text"]
                else:
                    text_content = text
            except json.JSONDecodeError:
                text_content = text

            # 【核心修复】：必须在文本末尾加上 EOS token
            # 这样模型才知道这句话讲完了，要开始预测下一句或者 padding 了
            self.examples.append(text_content + self.tokenizer.pad_token)

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
        # Remove batch dimension if present: [1, seq_len] -> [seq_len]
        if input_ids.dim() > 1:
            input_ids = input_ids.squeeze(0)
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
    gradient_accumulation_steps=1,
    accelerator=None,
    warmup_ratio=0.0,
):
    dataset = TextDataset(train_file, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    optimizer = AdamW(model.parameters(), lr=lr)

    # Calculate total steps for scheduler
    steps_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    optimizer_steps_per_epoch = (
        steps_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    # Calculate warmup steps
    warmup_steps = int(total_optimizer_steps * warmup_ratio)

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
        logger_name="pretrain",
    )


if __name__ == "__main__":
    # Example usage (commented out to avoid import errors if run directly without context)
    pass
