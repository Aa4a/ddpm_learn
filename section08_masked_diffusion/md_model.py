# -*- coding: utf-8 -*-
"""
Section 08 - Masked Diffusion 核心模块

对照：
  连续 DDPM:  x_t = √ᾱ x0 + √(1-ᾱ) ε ，网络预测 ε
  Mask 扩散:  以概率 t 把 token 换成 <mask>，网络预测被抹掉的原 token

骨干：双向 Transformer Encoder（无 Causal Mask）—— 这是相对 §06 自回归 Decoder 的关键差异。
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
class MaskDiffusionConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 64
    pad_idx: int = 0
    mask_idx: int = 4


# ---------------------------------------------------------------------------
# 位置编码 + Encoder 块
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None):
        # key_padding_mask: (B, L)，True = pad（PyTorch 约定：True 表示忽略）
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class MaskPredictor(nn.Module):
    """双向 Transformer：给定部分 mask 的序列，预测每个位置的原 token 分布。"""

    def __init__(self, cfg: MaskDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.pe = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) token ids（可含 <mask>）
        return logits: (B, L, V)
        """
        pad_mask = x == self.cfg.pad_idx  # True = pad
        h = self.embed(x) * math.sqrt(self.cfg.d_model)
        h = self.pe(h)
        for layer in self.layers:
            h = layer(h, key_padding_mask=pad_mask)
        h = self.norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 正向 Mask 过程
# ---------------------------------------------------------------------------

def q_sample(
    x0: torch.Tensor,
    mask_ratio: torch.Tensor | float,
    mask_idx: int,
    pad_idx: int,
    protect_special: bool = True,
    bos_idx: int | None = None,
    eos_idx: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    正向过程：每个「可破坏」位置独立以概率 t 变成 <mask>。

    x0: (B, L)
    mask_ratio: 标量或 (B,) —— 对应连续时间 t ∈ (0,1]
    返回:
      xt: 被 mask 后的序列
      mask: bool (B, L)，True = 本步被换成了 mask（也是训练要算 loss 的位置）
    """
    B, L = x0.shape
    device = x0.device
    if isinstance(mask_ratio, float):
        t = torch.full((B,), mask_ratio, device=device)
    else:
        t = mask_ratio.to(device).float()

    # 可破坏位置：非 pad；可选保护 bos/eos（教学默认保护，避免句界被抹掉）
    eligible = x0 != pad_idx
    if protect_special:
        if bos_idx is not None:
            eligible = eligible & (x0 != bos_idx)
        if eos_idx is not None:
            eligible = eligible & (x0 != eos_idx)

    # 每个位置独立 Bernoulli(t)
    rand = torch.rand(B, L, device=device)
    will_mask = (rand < t[:, None]) & eligible

    xt = x0.clone()
    xt[will_mask] = mask_idx
    return xt, will_mask


def sample_mask_ratio(batch_size: int, device: torch.device, eps: float = 1e-3) -> torch.Tensor:
    """t ~ U[eps, 1]，避免 t=0 时无 mask 导致 loss 为空。"""
    return torch.empty(batch_size, device=device).uniform_(eps, 1.0)


# ---------------------------------------------------------------------------
# 训练损失（Masked Diffusion / LLaDA 风格）
# ---------------------------------------------------------------------------

def masked_diffusion_loss(
    model: MaskPredictor,
    x0: torch.Tensor,
    pad_idx: int,
    mask_idx: int,
    bos_idx: int,
    eos_idx: int,
    use_t_weight: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    只在 mask 位做 CrossEntropy。

    use_t_weight=True（默认，贴近 LLaDA/MDLM）:
      L = E_t [ (1/t) * mean CE(logits[mask], x0[mask]) ]
    use_t_weight=False（教学小语料更稳）:
      L = mean CE on masked positions
    """
    B = x0.shape[0]
    device = x0.device
    t = sample_mask_ratio(B, device)
    xt, mask = q_sample(
        x0, t, mask_idx, pad_idx, protect_special=True, bos_idx=bos_idx, eos_idx=eos_idx
    )

    logits = model(xt)  # (B, L, V)

    if mask.any():
        ce = F.cross_entropy(logits[mask], x0[mask], reduction="none")
        if use_t_weight:
            batch_idx = torch.arange(B, device=device)[:, None].expand_as(mask)[mask]
            weights = 1.0 / t[batch_idx].clamp_min(1e-3)
            loss = (ce * weights).mean()
        else:
            loss = ce.mean()
    else:
        loss = logits.sum() * 0.0

    stats = {
        "mask_ratio_mean": float(t.mean().item()),
        "n_masked": int(mask.sum().item()),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# 采样：迭代 Unmask（低置信 remask）
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(
    model: MaskPredictor,
    vocab_size: int,
    seq_len: int,
    mask_idx: int,
    pad_idx: int,
    bos_idx: int,
    eos_idx: int,
    steps: int = 8,
    temperature: float = 1.0,
    device: torch.device | None = None,
    prompt_ids: list[int] | None = None,
) -> torch.Tensor:
    """
    从「几乎全 mask」出发，多步并行预测并逐步揭开。

    每一步：
      1. 对当前仍为 mask 的位置预测 token 分布
      2. 采样 / argmax 得到候选
      3. 只保留置信度最高的一部分，其余 remask（低置信 remasking）

    prompt_ids: 若提供，这些位置始终可见（条件生成 / SFT 直觉的迷你版）。
    """
    device = device or next(model.parameters()).device
    model.eval()

    # 初始化：全 mask，首尾可放 bos/eos 作为句界提示
    x = torch.full((1, seq_len), mask_idx, dtype=torch.long, device=device)
    x[0, 0] = bos_idx
    x[0, -1] = eos_idx

    known = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    known[0, 0] = True
    known[0, -1] = True

    if prompt_ids is not None:
        plen = min(len(prompt_ids), seq_len)
        x[0, :plen] = torch.tensor(prompt_ids[:plen], device=device)
        known[0, :plen] = True

    # 还需要填的内容位置数
    n_unknown = int((~known).sum().item())
    if n_unknown == 0:
        return x

    # 每步大约揭开 n_unknown / steps 个位置
    for step in range(steps):
        still_mask = ~known
        if not still_mask.any():
            break

        logits = model(x)  # (1, L, V)
        logits = logits / max(temperature, 1e-5)

        # 禁止采样出 pad / mask，避免退化
        logits[..., pad_idx] = -1e9
        logits[..., mask_idx] = -1e9

        if temperature <= 1e-6:
            conf, pred = torch.softmax(logits, dim=-1).max(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(1, -1)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

        conf = conf.masked_fill(~still_mask, -1.0)

        remaining = int(still_mask.sum().item())
        n_reveal = max(1, math.ceil(remaining / (steps - step)))
        n_reveal = min(n_reveal, remaining)

        topk = torch.topk(conf.view(-1), k=n_reveal)
        reveal_idx = topk.indices

        flat_pred = pred.view(-1)
        flat_x = x.view(-1)
        flat_known = known.view(-1)

        flat_x[reveal_idx] = flat_pred[reveal_idx]
        flat_known[reveal_idx] = True
        flat_x[~flat_known] = mask_idx

    return x


@torch.no_grad()
def decode_sample(ids: torch.Tensor, id_to_token: dict[int, str], skip_special: bool = True) -> str:
    special = {"<pad>", "<bos>", "<eos>", "<unk>", "<mask>"}
    toks = []
    for i in ids.view(-1).tolist():
        tok = id_to_token.get(i, "<unk>")
        if skip_special and tok in special:
            continue
        toks.append(tok)
    return " ".join(toks)
