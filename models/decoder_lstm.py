"""models/decoder_lstm.py — §7
BahdanauAttention + LSTMDecoder (Hướng A1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import (
    PROJECTION_DIM, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    MAX_ANSWER_LEN, PAD_ID, SOS_ID, EOS_ID,
)


class BahdanauAttention(nn.Module):
    """
    Cơ chế Bahdanau Attention — tính trọng số chú ý của LSTM hidden state
    lên chuỗi fusion_memory để tổng hợp vector ngữ cảnh tại mỗi bước sinh.
    """

    def __init__(self, hidden_dim: int, proj_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(proj_dim,   attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim,   1,         bias=False)

    def forward(
        self,
        h:      torch.Tensor,   # (batch, hidden_dim)
        memory: torch.Tensor,   # (batch, seq, proj_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_exp   = self.W_h(h).unsqueeze(1)
        s_exp   = self.W_s(memory)
        score   = self.v(torch.tanh(h_exp + s_exp)).squeeze(-1)
        alpha   = torch.softmax(score, dim=-1)
        context = (alpha.unsqueeze(-1) * memory).sum(dim=1)
        return context, alpha


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder 2 lớp có hỗ trợ Weight Tying với AnswerVocab.
    Hỗ trợ Teacher Forcing (train) và Beam Search (inference).
    """

    def __init__(
        self,
        vocab_size: int,
        proj_dim:   int   = PROJECTION_DIM,
        emb_dim:    int   = 768,
        hidden_dim: int   = LSTM_HIDDEN_DIM,
        num_layers: int   = LSTM_NUM_LAYERS,
        dropout:    float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.init_h    = nn.Linear(proj_dim, hidden_dim)
        self.init_c    = nn.Linear(proj_dim, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim, proj_dim)
        self.lstm      = nn.LSTM(
            input_size=emb_dim + proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        # Weight Tying
        self.hidden_to_emb = nn.Linear(hidden_dim, emb_dim)
        self.output_proj   = nn.Linear(emb_dim, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, fusion_cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.tanh(self.init_h(fusion_cls)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.tanh(self.init_c(fusion_cls)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0

    def forward(
        self,
        fusion_memory: torch.Tensor,
        fusion_cls:    torch.Tensor,
        answer_ids:    torch.Tensor,
    ) -> torch.Tensor:
        """Teacher Forcing — trả về logits (batch, T-1, vocab_size)."""
        _, seq_len = answer_ids.shape
        h, c = self._init_hidden(fusion_cls)
        logits = []
        for t in range(seq_len - 1):
            emb             = self.dropout(self.embedding(answer_ids[:, t]))
            context, _      = self.attention(h[-1], fusion_memory)
            lstm_in         = torch.cat([emb, context], dim=-1).unsqueeze(1)
            out, (h, c)     = self.lstm(lstm_in, (h, c))
            out_dropped     = self.dropout(out.squeeze(1))
            emb_space       = self.hidden_to_emb(out_dropped)
            logits.append(self.output_proj(emb_space))
        return torch.stack(logits, dim=1)

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
        h, c   = self._init_hidden(fusion_cls)
        beams  = [(0.0, [SOS_ID], h, c)]

        for _ in range(max_len):
            candidates = []
            for log_prob, tokens, h_b, c_b in beams:
                if tokens[-1] == EOS_ID:
                    candidates.append((log_prob, tokens, h_b, c_b))
                    continue
                token_tensor    = torch.tensor([tokens[-1]], device=device)
                emb             = self.embedding(token_tensor)
                context, _      = self.attention(h_b[-1], fusion_memory)
                lstm_in         = torch.cat([emb, context], dim=-1).unsqueeze(1)
                out, (h_new, c_new) = self.lstm(lstm_in, (h_b, c_b))
                emb_space       = self.hidden_to_emb(out.squeeze(1))
                logit           = self.output_proj(emb_space)
                log_probs       = F.log_softmax(logit, dim=-1).squeeze(0)
                topk_lp, topk_ids = log_probs.topk(beam_size)
                for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidates.append((log_prob + lp, tokens + [tid], h_new, c_new))

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]
            if all(b[1][-1] == EOS_ID for b in beams):
                break

        return beams[0][1]
