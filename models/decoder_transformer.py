"""models/decoder_transformer.py — §8
TransformerDecoder (Hướng A2) với Weight Tying và Beam Search.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import (
    PROJECTION_DIM, TRANSFORMER_NUM_LAYERS, TRANSFORMER_NUM_HEADS,
    TRANSFORMER_FFN_DIM, TRANSFORMER_DROPOUT,
    MAX_ANSWER_LEN, PAD_ID, SOS_ID, EOS_ID,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder sinh câu trả lời hỗ trợ Weight Tying.
    """

    def __init__(
        self,
        vocab_size:  int,
        proj_dim:    int   = PROJECTION_DIM,
        emb_dim:     int   = 768,
        num_layers:  int   = TRANSFORMER_NUM_LAYERS,
        num_heads:   int   = TRANSFORMER_NUM_HEADS,
        ffn_dim:     int   = TRANSFORMER_FFN_DIM,
        dropout:     float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.proj_dim   = proj_dim
        self.vocab_size = vocab_size

        self.embedding  = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.pos_enc    = PositionalEncoding(emb_dim, dropout)
        self.input_proj = nn.Linear(emb_dim, proj_dim) if emb_dim != proj_dim else nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=proj_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Weight Tying
        self.hidden_to_emb = nn.Linear(proj_dim, emb_dim)
        self.output_proj   = nn.Linear(emb_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        fusion_memory: torch.Tensor,
        fusion_cls:    torch.Tensor,
        answer_ids:    torch.Tensor,
    ) -> torch.Tensor:
        """Teacher Forcing — trả về logits (batch, T-1, vocab_size)."""
        tgt_in      = answer_ids[:, :-1]
        seq_len     = tgt_in.size(1)
        device      = tgt_in.device
        tgt_emb     = self.input_proj(self.pos_enc(self.embedding(tgt_in)))
        causal_mask = self._causal_mask(seq_len, device)
        pad_mask    = (tgt_in == PAD_ID)
        out         = self.transformer(
            tgt=tgt_emb, memory=fusion_memory,
            tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask,
        )
        emb_space = self.hidden_to_emb(out)
        return self.output_proj(emb_space)

    @torch.no_grad()
    def beam_search(
        self,
        fusion_memory: torch.Tensor,
        fusion_cls:    torch.Tensor,
        beam_size: int = 3,
        max_len:   int = MAX_ANSWER_LEN,
    ) -> list[int]:
        """Beam Search inference — trả về chuỗi ID tốt nhất."""
        device = fusion_memory.device
        beams  = [(0.0, [SOS_ID])]

        for _ in range(max_len):
            candidates = []
            for log_prob, tokens in beams:
                if tokens[-1] == EOS_ID:
                    candidates.append((log_prob, tokens))
                    continue
                tgt_tensor  = torch.tensor(tokens, device=device).unsqueeze(0)
                tgt_emb     = self.input_proj(self.pos_enc(self.embedding(tgt_tensor)))
                causal_mask = self._causal_mask(tgt_tensor.size(1), device)
                out         = self.transformer(tgt=tgt_emb, memory=fusion_memory, tgt_mask=causal_mask)
                emb_space   = self.hidden_to_emb(out[:, -1, :])
                logit       = self.output_proj(emb_space)
                log_probs   = F.log_softmax(logit, dim=-1).squeeze(0)
                topk_lp, topk_ids = log_probs.topk(beam_size)
                for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidates.append((log_prob + lp, tokens + [tid]))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(b[1][-1] == EOS_ID for b in beams):
                break

        return beams[0][1]
