"""training/loss.py — §10.1
LabelSmoothedCrossEntropy: hàm mất mát với Label Smoothing, bỏ qua PAD.
"""
import torch
import torch.nn as nn

from configs.config import LABEL_SMOOTHING, PAD_ID


class LabelSmoothedCrossEntropy(nn.Module):
    """
    CrossEntropyLoss tích hợp Label Smoothing của PyTorch.
    Tối ưu VRAM do không sinh ra ma trận tính toán nháp.
    """

    def __init__(
        self,
        vocab_size:   int,
        epsilon:      float = LABEL_SMOOTHING,
        ignore_index: int   = PAD_ID,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=epsilon,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits:  torch.Tensor,   # (batch, T-1, vocab_size)
        targets: torch.Tensor,   # (batch, T-1)
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)
        return self.loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
