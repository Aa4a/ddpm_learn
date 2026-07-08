# -*- coding: utf-8 -*-
"""
Section 06.4 - 位置编码：告诉模型「词在第几位」

Embedding 本身不带顺序信息（"i love you" 打乱 embedding 后注意力无法区分位置）。
Transformer 用正弦/余弦位置编码注入位置——与 Section 04 DDPM 的时间嵌入同源。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from transformer import PositionalEncoding

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def main():
    print("=" * 60)
    print("Section 06.4 - 位置编码")
    print("=" * 60)

    d_model = 32
    max_len = 20
    pe_layer = PositionalEncoding(d_model, max_len=max_len, dropout=0.0)

    # 取 pe buffer: (1, max_len, d_model)
    pe = pe_layer.pe[0].detach()  # (max_len, d_model)

    print(f"""
【为什么需要？】
  自注意力对输入顺序不敏感：如果只给词向量，模型不知道 "i love" 和 "love i" 的区别。
  解决：word_embedding + positional_encoding

【公式】（与 Section 04 时间嵌入相同结构）
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

  pos = 词在句子中的位置（0, 1, 2, ...）
  d   = d_model（本 demo 用 {d_model}）
""")

    print("【前 3 个位置、前 8 维数值（部分）】")
    for pos in range(3):
        vals = pe[pos, :8].tolist()
        s = ", ".join(f"{v:+.3f}" for v in vals)
        print(f"  pos={pos}: [{s}, ...]")

    # 演示：加 PE 前后
    fake_embed = torch.zeros(1, 5, d_model)  # 5 个词，embedding 全 0
    out = pe_layer(fake_embed)
    print("\n【演示】embedding 全 0 时，加 PE 后每个位置向量不同 → 模型能区分位置")
    print(f"  pos=0 向量前 4 维: {out[0, 0, :4].tolist()}")
    print(f"  pos=1 向量前 4 维: {out[0, 1, :4].tolist()}")

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pe[:max_len].T.numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("Position in sentence")
    ax.set_ylabel("Embedding dimension")
    ax.set_title(f"Sinusoidal Positional Encoding (d_model={d_model})")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "04_positional_encoding.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n曲线图已保存: {path}")
    print("""
【与 DDPM 的联系】
  Section 04:  t = 扩散时间步  →  正弦时间嵌入  →  告诉 U-Net「现在去噪到哪一步」
  Section 06:  pos = 词位置    →  正弦位置编码  →  告诉 Transformer「这是第几个词」

下一步: python 05_multihead_and_mask.py
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
