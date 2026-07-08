# -*- coding: utf-8 -*-
"""
Section 06.3 - 注意力机制直觉（单头、手工小例子）

在引入完整 Transformer 之前，先用 3 个英文词的小例子理解：
  「查询 Q 和所有键 K 算相似度 → softmax 得权重 → 加权求和 V」

运行后会保存注意力权重热力图到 figures/03_attention_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def scaled_dot_product_attention(Q, K, V):
    """Q: (L_q, d_k), K: (L_k, d_k), V: (L_k, d_v)"""
    d_k = Q.size(-1)
    scores = Q @ K.T / (d_k ** 0.5)          # (L_q, L_k)
    weights = F.softmax(scores, dim=-1)       # 每行和为 1
    out = weights @ V                          # (L_q, d_v)
    return out, weights, scores


def main():
    print("=" * 60)
    print("Section 06.3 - 注意力直觉")
    print("=" * 60)

    words = ["i", "love", "you"]
    L = len(words)
    d = 4  # 很小的维度，方便手算理解

    print(f"\n【设定】3 个词: {words}，每个词用 {d} 维向量表示。")
    print("  下面用随机向量模拟 Embedding（真实训练里向量是学出来的）。")

    torch.manual_seed(42)
    X = torch.randn(L, d)  # 同时当作 Q、K、V（自注意力：三者同源）

    out, weights, scores = scaled_dot_product_attention(X, X, X)

    print("\n【Step 1】算相似度 scores = Q K^T / sqrt(d_k)")
    print("  scores[i,j] = 第 i 个词「关注」第 j 个词的程度（未归一化）")
    for i, w in enumerate(words):
        row = "  ".join(f"{scores[i, j].item():6.2f}" for j in range(L))
        print(f"    {w:5s} → [{row}]")

    print("\n【Step 2】对每一行做 softmax → 注意力权重（和为 1）")
    for i, w in enumerate(words):
        row = "  ".join(f"{weights[i, j].item():6.3f}" for j in range(L))
        print(f"    {w:5s} → [{row}]")

    print("\n【Step 3】用权重对 V 加权求和 → 输出向量")
    print(f"  输出 shape: {tuple(out.shape)}  （每个词得到一个新的上下文向量）")

    print("""
【直觉】
  - 翻译 "love" 时，模型可能更关注 "i" 和 "you"（主语、宾语）
  - 注意力 = 「当前词该看句子里的哪些位置」
  - Cross-Attention 里 Q 来自 Decoder，K/V 来自 Encoder（译时看原文）

【公式】（先记住形状，细节后面会反复出现）
  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V
""")

    # 热力图
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(weights.detach().numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(words)
    ax.set_yticklabels(words)
    ax.set_xlabel("Key（被看谁）")
    ax.set_ylabel("Query（谁在看）")
    ax.set_title("Self-Attention Weights (demo)")
    for i in range(L):
        for j in range(L):
            ax.text(j, i, f"{weights[i, j].item():.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "03_attention_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n热力图已保存: {path}")
    print("\n下一步: python 04_positional_encoding.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
