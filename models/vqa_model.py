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
