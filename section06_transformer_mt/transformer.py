# -*- coding: utf-8 -*-
"""
Section 06 - Transformer 机器翻译核心模块

实现「Attention Is All You Need」(Vaswani et al., 2017) 的 Encoder-Decoder：
  - 正弦位置编码
  - 多头自注意力 / 交叉注意力
  - Encoder / Decoder 堆叠
  - 训练用 Teacher Forcing，推理用自回归贪心或 Beam Search
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 128
    pad_idx: int = 0


# ---------------------------------------------------------------------------
# 位置编码
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """正弦位置编码：PE(pos, 2i) = sin(pos / 10000^{2i/d}), PE(pos, 2i+1) = cos(...)"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model)，方便与 batch 广播
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 多头注意力
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """缩放点积多头注意力。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = query.size(0)

        # (B, L, d_model) -> (B, n_heads, L, d_k)
        Q = self.w_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # scores: (B, n_heads, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)  # (B, n_heads, L_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# 前馈网络
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder 层
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        # Pre-LN 风格也可，这里用论文原始 Post-LN
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, memory_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


# ---------------------------------------------------------------------------
# 完整 Transformer
# ---------------------------------------------------------------------------

class TransformerMT(nn.Module):
    """Encoder-Decoder Transformer，用于机器翻译。"""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_decoder_layers)
            ]
        )
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.cfg.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.cfg.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None,
        tgt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        src: (B, L_src)  源句子 token ids
        tgt: (B, L_tgt)  目标句子（已右移，含 <bos>，不含最后一步预测目标）
        返回 logits: (B, L_tgt, V_tgt)
        """
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.generator(out)


# ---------------------------------------------------------------------------
# Mask 工具
# ---------------------------------------------------------------------------

def make_pad_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """(B, L) -> (B, 1, 1, L)，1=有效，0=pad，可广播到注意力 scores。"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """下三角因果 mask：(1, 1, L, L)，防止看到未来 token。"""
    return torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Pad mask ∩ Causal mask。"""
    pad = make_pad_mask(tgt, pad_idx)  # (B, 1, 1, L)
    causal = make_causal_mask(tgt.size(1), tgt.device)  # (1, 1, L, L)
    return pad & causal.bool()


# ---------------------------------------------------------------------------
# 推理：贪心解码
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(
    model: TransformerMT,
    src: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_len: int = 64,
) -> torch.Tensor:
    """
    src: (1, L_src) 或 (B, L_src)
    返回: (B, L_out) 含 <bos>...<eos>
    """
    model.eval()
    device = src.device
    B = src.size(0)
    src_mask = make_pad_mask(src, pad_idx)
    memory = model.encode(src, src_mask)

    ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx)
        out = model.decode(ys, memory, tgt_mask, src_mask)
        logits = model.generator(out[:, -1])  # (B, V)
        next_tok = logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        # 已结束的样本继续填 eos，避免改变长度逻辑
        next_tok = torch.where(finished.unsqueeze(1), torch.full_like(next_tok, eos_idx), next_tok)
        ys = torch.cat([ys, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == eos_idx)
        if finished.all():
            break

    return ys
