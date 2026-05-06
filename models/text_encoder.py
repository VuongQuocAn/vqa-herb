"""models/text_encoder.py — §5
TextEncoder dựa trên PhoBERT-base-v2.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from configs.config import PHOBERT_MODEL_NAME


class TextEncoder(nn.Module):
    """
    Bộ mã hóa văn bản dựa trên PhoBERT-base-v2.
    Mặc định toàn bộ trọng số bị đóng băng trong giai đoạn 1.
    """

    def __init__(self, model_name: str = PHOBERT_MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
            token_embeddings : (batch, seq_len, 768)
            cls_embedding    : (batch, 768)
        """
        output           = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state
        cls_embedding    = output.last_hidden_state[:, 0, :]
        return token_embeddings, cls_embedding

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n: int) -> None:
        """Mở khóa n lớp transformer cuối cùng và pooler."""
        self.freeze()
        for layer in self.model.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        if hasattr(self.model, "pooler") and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Đã mở khóa {n} lớp cuối PhoBERT — {trainable:,} tham số có thể huấn luyện.")

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
