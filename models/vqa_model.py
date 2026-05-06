"""models/vqa_model.py — §9
VQAModel: gắn kết toàn bộ ImageEncoder, TextEncoder, Fusion và Decoder.
"""
import torch
import torch.nn as nn

from models.image_encoder      import ImageEncoder
from models.text_encoder       import TextEncoder
from models.fusion             import CoAttentionFusion
from models.decoder_lstm       import LSTMDecoder
from models.decoder_transformer import TransformerDecoder


class VQAModel(nn.Module):
    """
    Mô hình VQA hoàn chỉnh.

    Luồng dữ liệu:
        ảnh        → ImageEncoder   → patch_embeddings, cls_img
        câu hỏi   → TextEncoder    → token_embeddings, cls_txt
        patch + token → CoAttentionFusion → fusion_memory, fusion_cls
        fusion_memory + fusion_cls → Decoder → logits
    """

    def __init__(self, vocab_size: int, decoder_type: str = "transformer"):
        super().__init__()
        assert decoder_type in ("lstm", "transformer"), \
            f"decoder_type phải là 'lstm' hoặc 'transformer', nhận: '{decoder_type}'"

        self.decoder_type  = decoder_type
        self.image_encoder = ImageEncoder()
        self.text_encoder  = TextEncoder()
        self.fusion        = CoAttentionFusion()

        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(vocab_size=vocab_size)
        else:
            self.decoder = TransformerDecoder(vocab_size=vocab_size)

    def forward(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        answer_ids:     torch.Tensor,
    ) -> torch.Tensor:
        """Teacher Forcing — trả về logits (batch, T-1, vocab_size)."""
        patch_emb, _              = self.image_encoder(pixel_values)
        token_emb, _              = self.text_encoder(input_ids, attention_mask)
        fusion_memory, fusion_cls = self.fusion(patch_emb, token_emb, attention_mask)
        return self.decoder(fusion_memory, fusion_cls, answer_ids)

    @torch.no_grad()
    def predict(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        beam_size: int = 3,
    ) -> list[int]:
        """Beam Search inference — trả về chuỗi ID token."""
        self.eval()
        patch_emb, _              = self.image_encoder(pixel_values)
        token_emb, _              = self.text_encoder(input_ids, attention_mask)
        fusion_memory, fusion_cls = self.fusion(patch_emb, token_emb, attention_mask)
        return self.decoder.beam_search(fusion_memory, fusion_cls, beam_size=beam_size)

    @torch.no_grad()
    def predict_with_attention(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        beam_size: int = 3,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Beam Search inference + trích xuất attention map.
        Trả về:
            pred_ids     : chuỗi ID token
            attn_map_2d  : (14, 14) — attention của câu hỏi lên các patch ảnh,
                           đã được chuẩn hóa về [0, 1].
        """
        self.eval()
        patch_emb, _ = self.image_encoder(pixel_values)
        token_emb, _ = self.text_encoder(input_ids, attention_mask)

        patch_proj = self.fusion.img_proj(patch_emb)
        token_proj = self.fusion.text_proj(token_emb)

        # Text → Image attention: câu hỏi nhìn vào vùng nào của ảnh?
        _, attn_weights = self.fusion.text_guided_img_attn(
            query=token_proj, key=patch_proj, value=patch_proj,
            need_weights=True, average_attn_weights=True,
        )
        # attn_weights: (1, seq_len, num_patches) → trung bình qua các token
        attn_map = attn_weights[0].mean(dim=0)           # (num_patches,) = 256

        # DINOv2-base: 224/16 = 14 → grid 14×14 = 196 patches
        # CoAttentionFusion chiếu 768→256 nên giữ nguyên num_patches
        num_patches = attn_map.shape[0]
        grid_size   = int(num_patches ** 0.5)            # 14 nếu num_patches=196
        if grid_size * grid_size == num_patches:
            attn_map_2d = attn_map.reshape(grid_size, grid_size)
        else:
            # fallback: cắt bớt cho vừa grid gần nhất
            gs = int(num_patches ** 0.5)
            attn_map_2d = attn_map[:gs * gs].reshape(gs, gs)

        # Chuẩn hóa về [0, 1]
        attn_map_2d = (attn_map_2d - attn_map_2d.min()) / (attn_map_2d.max() - attn_map_2d.min() + 1e-8)

        fusion_memory, fusion_cls = self.fusion(patch_emb, token_emb, attention_mask)
        pred_ids = self.decoder.beam_search(fusion_memory, fusion_cls, beam_size=beam_size)

        return pred_ids, attn_map_2d.cpu()

    # ── Freeze / Unfreeze helpers ────────────────────────────────────────────
    def freeze_encoders(self) -> None:
        self.image_encoder.freeze()
        self.text_encoder.freeze()

    def unfreeze_encoders_partial(self, n_img_blocks: int = 2, n_txt_layers: int = 2) -> None:
        self.image_encoder.unfreeze_last_n_blocks(n_img_blocks)
        self.text_encoder.unfreeze_last_n_layers(n_txt_layers)

    def count_params(self) -> dict:
        """Thống kê số tham số theo từng thành phần."""
        def _count(module):
            total     = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        parts = {
            "ImageEncoder": self.image_encoder,
            "TextEncoder":  self.text_encoder,
            "Fusion":       self.fusion,
            "Decoder":      self.decoder,
        }
        result = {}
        grand_total = grand_train = 0
        for name, mod in parts.items():
            t, tr = _count(mod)
            result[name] = {"total": t, "trainable": tr}
            grand_total += t
            grand_train += tr
        result["TOTAL"] = {"total": grand_total, "trainable": grand_train}
        return result
