# -*- coding: utf-8 -*-
"""
Section 08.4 - 训练目标：mask 位交叉熵

不训练真实网络，用「假 logits」演示：
  1. 只对 mask 位置算 CE
  2. 常见加权 1/t（掩码越多，单步信号越稀，需放大）
"""

import torch
import torch.nn.functional as F

from data_utils import build_vocab, setup_stdio, tokenize
from md_model import q_sample, sample_mask_ratio


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 08.4 - Masked Diffusion 训练目标")
    print("=" * 60)

    vocab = build_vocab()
    text = "i love you"
    x0 = torch.tensor([vocab.encode(tokenize(text))], dtype=torch.long)
    B, L = x0.shape
    V = len(vocab)

    torch.manual_seed(42)
    t = sample_mask_ratio(B, x0.device)
    xt, mask = q_sample(
        x0,
        t,
        mask_idx=vocab.mask_idx,
        pad_idx=vocab.pad_idx,
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
    )

    print(f"原句 x0 : {text}")
    print(f"采样 t  : {float(t[0]):.3f}")
    print(f"mask 位 : {mask[0].tolist()}")
    print(f"xt ids  : {xt[0].tolist()}")

    # 构造「半对半错」的假 logits，方便看 CE 行为
    logits = torch.randn(B, L, V) * 0.1
    # 在真实 token 上抬高一点分数 → 模拟「还行的模型」
    for b in range(B):
        for i in range(L):
            logits[b, i, x0[b, i]] += 2.0

    print("""
【损失怎么算】
  1. 网络输入 xt（含 <mask>），输出每个位置的词表 logits
  2. 只在 mask==True 的位置，对真实 x0 做 CrossEntropy
  3.（可选，贴近 LLaDA/MDLM）再乘权重 1/t 后取平均

  连续 DDPM:  L = ||ε - ε_θ(x_t, t)||²
  Mask 扩散:   L ≈ E_t[ (1/t) * CE(x0_mask, p_θ(·|xt)) ]
""")

    if mask.any():
        ce = F.cross_entropy(logits[mask], x0[mask], reduction="none")
        batch_idx = torch.arange(B)[:, None].expand_as(mask)[mask]
        w = 1.0 / t[batch_idx].clamp_min(1e-3)
        loss = (ce * w).mean()
        print(f"未加权 mean CE = {ce.mean().item():.4f}")
        print(f"加权后 loss    = {loss.item():.4f}  （本例 t≈{float(t[0]):.3f}）")
    else:
        print("本随机种子下没有 mask 位（极少见），请重跑。")

    print("""
【训练一步伪代码】
  t  ~ Uniform(ε, 1)
  xt, mask = q_sample(x0, t)          # 随机抹词
  logits   = BidirectionalTransformer(xt)
  loss     = weighted_CE(logits[mask], x0[mask], weight=1/t)
  loss.backward(); optimizer.step()

没有 Teacher Forcing 错位，也没有 Causal Mask ——
和 §06 翻译训练循环看起来很不一样，但 CE 预测离散 token 是同源的。
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
