# -*- coding: utf-8 -*-
"""
Section 06.5 - 多头注意力 + Pad Mask + 因果 Mask

在 06.3 单头注意力基础上：
  1. 多头：多组 Q/K/V 并行，看不同「子空间」
  2. Pad Mask：忽略补齐的 <pad>
  3. Causal Mask：Decoder 不能偷看未来的词
"""

import torch

from transformer import MultiHeadAttention, make_causal_mask, make_pad_mask, make_tgt_mask


def main():
    print("=" * 60)
    print("Section 06.5 - 多头注意力与 Mask")
    print("=" * 60)

    B, L, d_model, n_heads = 1, 5, 16, 4
    pad_idx = 0

    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    x = torch.randn(B, L, d_model)

    out = mha(x, x, x, mask=None)
    print(f"\n【1】多头自注意力")
    print(f"  输入 x:  {tuple(x.shape)}   (batch, 句长, d_model)")
    print(f"  输出:    {tuple(out.shape)}  （shape 不变）")
    print(f"  头数 n_heads={n_heads}，每头维度 d_k={d_model // n_heads}")
    print("  直觉：多个头分别学「语法」「指代」「词序」等不同模式，最后拼起来。")

    # Pad mask
    seq = torch.tensor([[1, 2, 3, 0, 0]])  # 后两个是 pad
    pad_mask = make_pad_mask(seq, pad_idx)
    print(f"\n【2】Pad Mask（忽略补齐位）")
    print(f"  句子 ids:     {seq[0].tolist()}")
    print(f"  pad_mask:     {pad_mask[0, 0, 0].tolist()}  （1=有效, 0=pad）")
    print("  作用：pad 位置不参与 attention，也不产生 loss。")

    # Causal mask
    causal = make_causal_mask(L, torch.device("cpu"))
    print(f"\n【3】Causal Mask（Decoder 不能看未来）")
    print("  下三角矩阵，位置 i 只能 attend 到 j<=i：")
    tri = causal[0, 0].int()
    for i in range(L):
        row = " ".join(str(tri[i, j].item()) for j in range(L))
        print(f"    pos {i}: [{row}]")
    print("  训练译文中：预测第 3 个字时，只能看到前 2 个字，不能偷看第 4 个字。")

    tgt = torch.tensor([[1, 2, 3, 4, 0]])
    tgt_mask = make_tgt_mask(tgt, pad_idx)
    print(f"\n【4】Decoder 实际用的是 Pad Mask AND Causal Mask 的交集")

    print("""
【三种注意力在本项目中的分工】
  Encoder Self-Attn:  Q,K,V 都来自英文  →  理解原文
  Decoder Self-Attn:  Q,K,V 都来自已生成的中文 + Causal Mask
  Cross-Attn:        Q 来自中文，K,V 来自 Encoder 输出（memory）

下一步: python 06_model_assembly.py
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
