"""models/fusion.py — §6
Co-attention Fusion Module: hợp nhất đặc trưng ảnh và văn bản.
"""
import torch
import torch.nn as nn

from configs.config import IMAGE_ENCODER_DIM, TEXT_ENCODER_DIM, PROJECTION_DIM


class CoAttentionFusion(nn.Module):
    """
    Module hợp nhất thông tin ảnh và văn bản theo cơ chế Co-attention.

    Bước 1 – Linear Projection: Ép chiều 768 → 256.
    Bước 2 – Image-guided Text Attention: patch làm Query, token làm Key/Value.
    Bước 3 – Text-guided Image Attention: token làm Query, patch làm Key/Value.
    Bước 4 – Tổng hợp bằng Max Pooling + Linear Projection.
    """

    def __init__(
        self,
        proj_dim:  int   = PROJECTION_DIM,
        num_heads: int   = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()

        self.img_proj  = nn.Linear(IMAGE_ENCODER_DIM, PROJECTION_DIM)
        self.text_proj = nn.Linear(TEXT_ENCODER_DIM, PROJECTION_DIM)

        self.img_guided_text_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.text_guided_img_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_text_attn = nn.LayerNorm(proj_dim)
        self.norm_img_attn  = nn.LayerNorm(proj_dim)

        self.fusion_proj = nn.Linear(proj_dim * 2, proj_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        patch_emb:      torch.Tensor,   # (batch, num_patches, 768)
        token_emb:      torch.Tensor,   # (batch, seq_len, 768)
        attention_mask: torch.Tensor,   # (batch, seq_len)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
            fusion_memory : (batch, num_patches, proj_dim)
            fusion_cls    : (batch, proj_dim)
        """
        patch_proj = self.img_proj(patch_emb)
        token_proj = self.text_proj(token_emb)

        pad_mask = (attention_mask == 0)

        text_attn, _ = self.img_guided_text_attn(
            query=patch_proj, key=token_proj, value=token_proj,
            key_padding_mask=pad_mask,
        )
        text_attn = self.norm_text_attn(text_attn + patch_proj)

        img_attn, _ = self.text_guided_img_attn(
            query=token_proj, key=patch_proj, value=patch_proj,
        )
        img_attn = self.norm_img_attn(img_attn + token_proj)

        fusion_memory = text_attn
        patch_global  = text_attn.max(dim=1)[0]
        token_global  = img_attn.max(dim=1)[0]
        combined      = torch.cat([patch_global, token_global], dim=-1)
        fusion_cls    = self.dropout(torch.tanh(self.fusion_proj(combined)))

        return fusion_memory, fusion_cls
