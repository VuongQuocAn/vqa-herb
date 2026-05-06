"""training/trainer.py — §10.3
train_one_epoch: Mixed Precision + Gradient Accumulation + Gradient Clipping.
"""
import time
import torch
from torch.amp import GradScaler, autocast

from configs.config import DEVICE, GRAD_ACCUMULATION, GRAD_CLIP_NORM


def _fmt_time(seconds: float) -> str:
    """Chuyển số giây thành chuỗi mm:ss hoặc hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    scaler:  GradScaler,
    epoch:   int,
    device:  torch.device = DEVICE,
) -> float:
    """
    Huấn luyện mô hình qua một epoch với:
      - Mixed Precision (fp16) để giảm VRAM và tăng tốc.
      - Gradient Accumulation để mô phỏng batch size lớn hơn.
      - Gradient Clipping để tránh gradient exploding.

    Trả về: loss trung bình trên toàn bộ epoch.
    """
    model.train()
    optimizer.zero_grad()
    total_loss  = 0.0
    num_updates = 0
    epoch_start = time.time()

    for step, batch in enumerate(loader, start=1):
        pixel_values   = batch["image"].to(device, non_blocking=True)
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        answer_ids     = batch["answer_ids"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.float16):
            logits      = model(pixel_values, input_ids, attention_mask, answer_ids)
            targets     = answer_ids[:, 1:]
            loss        = criterion(logits, targets)
            loss_scaled = loss / GRAD_ACCUMULATION

        scaler.scale(loss_scaled).backward()
        total_loss += loss.item()

        if step % GRAD_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=GRAD_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            num_updates += 1

        if step % (GRAD_ACCUMULATION * 50) == 0 or step == len(loader):
            elapsed      = time.time() - epoch_start
            eta_sec      = elapsed / step * (len(loader) - step)
            avg_loss     = total_loss / step
            current_lr   = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch} | Batch {step:>5}/{len(loader)} | "
                f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                f"ETA: {_fmt_time(eta_sec)}"
            )

    # Phần dư nếu tổng batch không chia hết cho GRAD_ACCUMULATION
    remainder = len(loader) % GRAD_ACCUMULATION
    if remainder != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            max_norm=GRAD_CLIP_NORM,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    epoch_time = time.time() - epoch_start
    avg_loss   = total_loss / len(loader)
    print(
        f"  Epoch {epoch} kết thúc | Loss TB: {avg_loss:.4f} | "
        f"Thời gian: {_fmt_time(epoch_time)} | Cập nhật: {num_updates} lần"
    )
    return avg_loss
