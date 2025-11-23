import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import copy
import json
from tqdm import tqdm
import os
import sys

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expects: [{"prompt": "...", "chosen": "...", "rejected": "..."}]
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        item = self.examples[i]
        prompt = (
            f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        )
        chosen = f"{item['chosen']}<|im_end|>"
        rejected = f"{item['rejected']}<|im_end|>"

        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected, add_special_tokens=False)

        # Helper to pad/truncate
        def prepare_sequence(p_ids, r_ids):
            input_ids = p_ids + r_ids
            labels = [-100] * len(p_ids) + r_ids

            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                labels = labels[: self.max_length]
            else:
                pad_len = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len
            return input_ids, labels

        chosen_input_ids, chosen_labels = prepare_sequence(prompt_ids, chosen_ids)
        rejected_input_ids, rejected_labels = prepare_sequence(prompt_ids, rejected_ids)

        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


def get_batch_logps(logits, labels, label_pad_token_id=-100):
    """
    Compute log probabilities of labels given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            "Logits (batch and sequence length dim) and labels must have the same shape."
        )

    labels = labels.clone()
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1)


def train_dpo(
    model,
    tokenizer,
    train_file,
    ref_model=None,
    beta=0.1,
    epochs=1,
    batch_size=2,
    lr=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    logger = Logger("dpo")
    logger.info(f"Starting DPO on {device}...")
    model.to(device)
    model.train()

    if ref_model is None:
        print(
            "No reference model provided, creating a copy of the model as reference..."
        )
        ref_model = copy.deepcopy(model)

    ref_model.to(device)
    ref_model.eval()

    dataset = DPODataset(train_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"DPO Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(loop):
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            # Concatenate chosen and rejected for efficient forward pass
            all_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
            all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

            # Policy forward
            all_logits = model(input_ids=all_input_ids).get("logits")
            all_logps = get_batch_logps(all_logits, all_labels)

            chosen_logps, rejected_logps = all_logps.chunk(2, dim=0)

            # Reference forward
            with torch.no_grad():
                ref_logits = ref_model(input_ids=all_input_ids).get("logits")
                ref_logps = get_batch_logps(ref_logits, all_labels)

            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

            # DPO Loss
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps

            logits = pi_logratios - ref_logratios
            losses = -F.logsigmoid(beta * logits)
            loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

            # Log metrics every 10 steps
            if i % 10 == 0:
                logger.log_metrics(
                    {"epoch": epoch + 1, "step": i, "loss": f"{loss.item():.4f}"}
                )

    logger.info("DPO finished.")
    return model
