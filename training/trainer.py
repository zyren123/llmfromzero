import torch
from tqdm import tqdm
from utils.logger import Logger


def train_model(
    model,
    dataloader,
    optimizer,
    accelerator,
    epochs,
    gradient_accumulation_steps,
    scheduler=None,
    logger_name="trainer",
):
    """
    Unified training loop for Pretraining and SFT.
    """
    logger = Logger(logger_name)
    logger.info(f"Starting training on {accelerator.device}...")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

    # Calculate effective batch size for logging
    # Note: batch_size here is per-device batch size from dataloader
    if hasattr(dataloader, "batch_size") and dataloader.batch_size:
        batch_size = dataloader.batch_size
    else:
        # Fallback if batch_size is not directly available (e.g. custom sampler)
        # Try to infer from first batch or just log "unknown"
        batch_size = "unknown"

    if isinstance(batch_size, int):
        effective_batch_size = (
            batch_size * gradient_accumulation_steps * accelerator.num_processes
        )
        logger.info(f"Effective Batch Size: {effective_batch_size}")

    model.train()

    # Calculate total steps for logging
    try:
        total_steps_per_epoch = len(dataloader)
        # Optimizer steps per epoch (accounting for accumulation)
        optimizer_steps_per_epoch = (
            total_steps_per_epoch + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        total_optimizer_steps = optimizer_steps_per_epoch * epochs
    except TypeError:
        # If dataloader doesn't have length
        total_optimizer_steps = 0

    global_step = 0
    optimizer_step = 0

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(loop):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # Use accelerator's accumulate context manager
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

                accelerator.backward(loss)

                # Only update optimizer when accumulation is complete
                if accelerator.sync_gradients:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

            global_step += 1

            # Calculate epoch progress
            if total_optimizer_steps > 0:
                epoch_decimal = optimizer_step / total_optimizer_steps
            else:
                epoch_decimal = 0.0

            current_lr = 0.0
            if scheduler:
                current_lr = scheduler.get_last_lr()[0]
            elif hasattr(optimizer, "param_groups"):
                current_lr = optimizer.param_groups[0]["lr"]

            loop.set_postfix(loss=loss.item(), lr=current_lr, opt_step=optimizer_step)

            # Log metrics every 10 steps
            if global_step % 10 == 0:
                metrics = {
                    "train_loss": loss.item(),
                    "epoch": epoch_decimal,
                    "step": global_step,
                    "optimizer_step": optimizer_step,
                    "learning_rate": current_lr,
                }
                accelerator.log(metrics)

                log_metrics = {
                    "epoch": f"{epoch_decimal:.4f}",
                    "step": global_step,
                    "opt_step": optimizer_step,
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.6f}",
                }
                logger.log_metrics(log_metrics)

    logger.info("Training finished.")
    return model
