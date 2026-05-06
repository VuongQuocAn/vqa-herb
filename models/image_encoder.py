"""models/image_encoder.py — §4
ImageEncoder dựa trên DINOv2-base (ViT-B/14).
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from configs.config import DINOV2_MODEL_NAME


class ImageEncoder(nn.Module):
    """
    Bộ mã hóa ảnh dựa trên DINOv2-base (ViT-B/14).
    Ảnh 224x224 được chia thành 256 patch (14x14 mỗi patch).
    Mặc định toàn bộ trọng số bị đóng băng trong giai đoạn 1.
    """

    def __init__(self, model_name: str = DINOV2_MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze()

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
            patch_embeddings : (batch, 256, 768)
            cls_embedding    : (batch, 768)
        """
        output  = self.model(pixel_values=pixel_values)
        hidden  = output.last_hidden_state
        cls_embedding    = hidden[:, 0, :]
        patch_embeddings = hidden[:, 1:, :]
        return patch_embeddings, cls_embedding

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Mở khóa n block cuối của ViT encoder và lớp LayerNorm đầu ra."""
        self.freeze()
        for block in self.model.encoder.layer[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.model.layernorm.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Đã mở khóa {n} block cuối DINOv2 — {trainable:,} tham số có thể huấn luyện.")

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
