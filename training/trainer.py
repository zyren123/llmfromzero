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
    logger = Logger(logger_name, is_main_process=accelerator.is_main_process)
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
    running_loss = 0.0
    running_gradnorm = 0.0
    gradnorm_count = 0
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
                    # 计算梯度范数（在优化器更新之前）
                    total_norm = 0.0
                    param_count = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    if param_count > 0:
                        total_norm = total_norm ** (1.0 / 2)
                        running_gradnorm += total_norm
                        gradnorm_count += 1

                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

            running_loss += loss.detach().float()

            global_step += 1

            # Log metrics every 10 steps (或者你可以设大一点，比如 50)
            if global_step % 10 == 0:
                # 1. 计算本地平均 loss (过去 10 步)
                local_avg_loss = running_loss / 10

                # 2. [关键] GATHER: 把所有设备的 local_avg_loss 收集起来
                # input:  scalar tensor (比如 rank0是 6.1, rank1是 6.3)
                # output: tensor([6.1, 6.3, ...]) 维度是 [num_processes]
                all_losses = accelerator.gather(local_avg_loss)

                # 3. 取全局平均
                # 这一步会自动处理 TPU/GPU 的同步
                global_avg_loss = all_losses.mean().item()

                # 计算平均梯度范数
                avg_gradnorm = 0.0
                if gradnorm_count > 0:
                    local_avg_gradnorm = running_gradnorm / gradnorm_count
                    # 收集所有设备的梯度范数
                    all_gradnorms = accelerator.gather(
                        torch.tensor(local_avg_gradnorm, device=accelerator.device)
                    )
                    avg_gradnorm = all_gradnorms.mean().item()
                    running_gradnorm = 0.0
                    gradnorm_count = 0

                # 重置累积器
                running_loss = 0.0

                # 获取学习率
                current_lr = 0.0
                if scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                elif hasattr(optimizer, "param_groups"):
                    current_lr = optimizer.param_groups[0]["lr"]

                # 4. 打印日志 (只在主进程打印)
                if accelerator.is_main_process:
                    loop.set_postfix(
                        loss=global_avg_loss,
                        gradnorm=avg_gradnorm,
                        lr=current_lr,
                        step=optimizer_step,
                    )

                    metrics = {
                        "train_loss": global_avg_loss,
                        "gradnorm": avg_gradnorm,
                        "epoch": optimizer_step / total_optimizer_steps
                        if total_optimizer_steps > 0
                        else 0,
                        "step": global_step,
                        "optimizer_step": optimizer_step,
                        "learning_rate": current_lr,
                    }
                    logger.log_metrics(metrics)
                    accelerator.log(metrics)

    logger.info("Training finished.")
    return model
