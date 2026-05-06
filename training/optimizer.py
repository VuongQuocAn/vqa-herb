"""training/optimizer.py — §10.2
build_optimizer_phase1 / phase2: AdamW + Cosine Warmup Scheduler.
"""
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from configs.config import LR_PHASE1, LR_PHASE2, WARMUP_STEPS


def build_optimizer_phase1(model: nn.Module, num_training_steps: int):
    """
    Giai đoạn 1 — Encoder đóng băng.
    Chỉ tối ưu Fusion và Decoder.
    """
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    trainable_params = [
        p for p in
        list(actual_model.fusion.parameters()) + list(actual_model.decoder.parameters())
        if p.requires_grad
    ]
    optimizer = AdamW(trainable_params, lr=LR_PHASE1, weight_decay=0.05,
                      betas=(0.9, 0.98), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def build_optimizer_phase2(model: nn.Module, num_training_steps: int):
    """
    Giai đoạn 2 — Mở khóa một phần Encoder.
    Encoder học rất chậm (LR * 0.1), Fusion/Decoder học nhanh hơn.
    """
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    encoder_params = [
        p for p in
        list(actual_model.image_encoder.parameters()) +
        list(actual_model.text_encoder.parameters())
        if p.requires_grad
    ]
    head_params = [
        p for p in
        list(actual_model.fusion.parameters()) + list(actual_model.decoder.parameters())
        if p.requires_grad
    ]
    param_groups = [
        {"params": encoder_params, "lr": LR_PHASE2 * 0.1},
        {"params": head_params,    "lr": LR_PHASE2},
    ]
    optimizer = AdamW(param_groups, weight_decay=0.05, betas=(0.9, 0.98), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS // 2,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler
