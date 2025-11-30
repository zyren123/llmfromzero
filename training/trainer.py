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
    # 仅在主进程初始化 Logger，防止多卡重复打印
    logger = Logger(logger_name, is_main_process=accelerator.is_main_process)
    logger.info(f"Starting training on {accelerator.device}...")
    logger.info(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")

    if hasattr(dataloader, "batch_size") and dataloader.batch_size:
        batch_size = dataloader.batch_size
    else:
        batch_size = "unknown"

    if isinstance(batch_size, int):
        effective_batch_size = (
            batch_size * gradient_accumulation_steps * accelerator.num_processes
        )
        logger.info(f"Effective Batch Size: {effective_batch_size}")

    model.train()

    # Calculate total steps
    try:
        total_steps_per_epoch = len(dataloader)
        optimizer_steps_per_epoch = (
            total_steps_per_epoch + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        total_optimizer_steps = optimizer_steps_per_epoch * epochs
    except TypeError:
        total_optimizer_steps = 0

    global_step = 0
    optimizer_step = 0
    
    # 累积变量
    running_loss = 0.0
    current_gradnorm = 0.0
    
    # 定义日志打印间隔
    log_interval = 10 

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not accelerator.is_main_process)
        
        for i, batch in enumerate(loop):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # 计算梯度范数 (仅在更新参数时计算)
                    total_norm = 0.0
                    param_count = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    if param_count > 0:
                        current_gradnorm = total_norm ** (1.0 / 2)

                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1

            # --- [修改点 1] ---
            # 直接累加当前卡的 loss.item()。
            # accelerate 的 loss 通常已经是 Mean Reduction (例如 8.0)
            # 我们不需要在这里除以 accumulation_steps，因为我们最后是求一段时间内的平均值
            running_loss += loss.item()
            
            global_step += 1

            # --- [修改点 2] ---
            # 每 log_interval 步打印一次日志
            if global_step % log_interval == 0:
                
                # --- [修改点 3] ---
                # 仅在主进程处理日志逻辑
                # 不再使用 gather，彻底避开多卡求和导致 Loss 翻倍的问题
                if accelerator.is_main_process:
                    # 计算过去 log_interval 步的平均 Loss
                    # 例如：(8.0 + 8.1 + ... + 7.9) / 10 = 8.0
                    avg_loss = running_loss / log_interval
                    
                    # 获取当前学习率
                    current_lr = 0.0
                    if scheduler:
                        current_lr = scheduler.get_last_lr()[0]
                    elif hasattr(optimizer, "param_groups"):
                        current_lr = optimizer.param_groups[0]["lr"]

                    # 更新进度条
                    loop.set_postfix(
                        loss=avg_loss,
                        gradnorm=current_gradnorm,
                        lr=current_lr,
                        step=optimizer_step,
                    )

                    # 记录日志
                    metrics = {
                        "train_loss": avg_loss,
                        "gradnorm": current_gradnorm,
                        "epoch": optimizer_step / total_optimizer_steps if total_optimizer_steps > 0 else 0,
                        "step": global_step,
                        "optimizer_step": optimizer_step,
                        "learning_rate": current_lr,
                    }
                    logger.log_metrics(metrics)
                    accelerator.log(metrics)

                # 重置累积器 (所有进程都要重置，保持逻辑一致)
                running_loss = 0.0

    logger.info("Training finished.")
    return model