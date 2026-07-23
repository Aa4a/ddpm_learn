# -*- coding: utf-8 -*-
"""
Section 08.2 - 正向过程：随机 Mask

对应连续 DDPM 的 q(x_t | x0)。
这里：t = 掩码比例，每个可破坏位置独立变成 <mask>。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_utils import build_vocab, ids_to_readable, setup_stdio, tokenize
from md_model import q_sample

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 08.2 - 正向 Mask 过程")
    print("=" * 60)

    vocab = build_vocab()
    text = "attention is all you need"
    ids = vocab.encode(tokenize(text), add_bos_eos=True)
    x0 = torch.tensor([ids], dtype=torch.long)

    print(f"\n原句: {text}")
    print(f"ids : {ids}")
    print(f"可读: {ids_to_readable(ids, vocab)}")

    print("\n【不同掩码比例 t 下的 xt】（bos/eos 默认保护）")
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    torch.manual_seed(0)
    for t in ratios:
        xt, mask = q_sample(
            x0,
            mask_ratio=t,
            mask_idx=vocab.mask_idx,
            pad_idx=vocab.pad_idx,
            bos_idx=vocab.bos_idx,
            eos_idx=vocab.eos_idx,
        )
        readable = ids_to_readable(xt[0], vocab)
        n = int(mask.sum().item())
        print(f"  t={t:.2f}  mask数={n:2d}  →  {readable}")
        rows.append((t, xt[0].tolist(), mask[0].tolist()))

    print("""
【公式直觉】
  对每个非特殊位置 i：
    x_t[i] = <mask>    以概率 t
    x_t[i] = x0[i]     以概率 1-t

  t=0  → 句子完好（无信息破坏）
  t=1  → 内容位全是 <mask>（最大「噪声」）

  对照 DDPM：
    t 大 ↔ ᾱ_t 小 ↔ 更接近纯噪声
""")

    # 可视化：一行一个 t，颜色标出 mask
    fig, axes = plt.subplots(len(ratios), 1, figsize=(10, 6), sharex=True)
    tokens0 = [vocab.id_to_token[i] for i in ids]
    for ax, (t, xt_ids, mflags) in zip(axes, rows):
        colors = ["#e74c3c" if m else "#3498db" for m in mflags]
        labels = [vocab.id_to_token[i] for i in xt_ids]
        ax.bar(range(len(labels)), [1] * len(labels), color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticks([])
        ax.set_ylabel(f"t={t:.2f}", fontsize=10)
        ax.set_xlim(-0.5, len(labels) - 0.5)
    axes[0].set_title("Forward masking (red = <mask>, blue = kept)")
    fig.tight_layout()
    out = FIG / "02_forward_masking.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"已保存图: {out}")
    print("原句 token 序列:", tokens0)
    print("=" * 60)


if __name__ == "__main__":
    main()
