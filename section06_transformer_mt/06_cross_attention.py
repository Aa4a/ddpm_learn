# -*- coding: utf-8 -*-
"""
Section 06.6 - Cross-Attention（交叉注意力）专节

机器翻译的核心桥梁：Decoder 写中文时，如何「回头查」Encoder 对英文的理解？

本脚本在 06.3（自注意力公式）和 06.5（多头 + Mask）之后，专门讲：
  1. Self-Attn vs Cross-Attn：Q/K/V 各从哪来
  2. 译「i love you → 我爱你」时，每个中文字关注哪些英文词
  3. shape 追踪：(L_tgt, L_src) 对齐矩阵
  4. 源码对应：DecoderLayer.cross_attn(x, memory, memory, ...)
  5. 源句 Pad Mask：英文补齐位不能被 attend

运行后保存 figures/06_cross_attention_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from transformer import MultiHeadAttention, make_pad_mask

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (L_q, d_k), K: (L_k, d_k), V: (L_k, d_v), mask: (L_q, L_k) 或 None"""
    d_k = Q.size(-1)
    scores = Q @ K.T / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = weights @ V
    return out, weights, scores


def print_attention_matrix(query_labels, key_labels, weights, title):
    print(f"\n{title}")
    header = "       " + "  ".join(f"{k:>6s}" for k in key_labels)
    print(header)
    for i, ql in enumerate(query_labels):
        row = "  ".join(f"{weights[i, j].item():6.3f}" for j in range(len(key_labels)))
        print(f"  {ql:5s} → [{row}]")


def main():
    print("=" * 60)
    print("Section 06.6 - Cross-Attention（交叉注意力）")
    print("=" * 60)

    print("""
【Cross-Attention 解决什么问题？】

  Encoder 读完英文，得到 memory（每个英文位置的上下文向量）。
  Decoder 写中文时，不能「闭着眼瞎写」——每写一个中文字，都要问：
    「原文里哪个位置和我现在要写的内容最相关？」

  Cross-Attention 就是这道「查原文」的接口。

【与 Self-Attention 对比】（公式完全相同，只是 Q/K/V 来源不同）

  ┌─────────────────┬──────────────────┬──────────────────┐
  │                 │ Self-Attention   │ Cross-Attention  │
  ├─────────────────┼──────────────────┼──────────────────┤
  │ Q（谁在问）      │ 本句自己          │ Decoder（译文）   │
  │ K（被谁索引）    │ 本句自己          │ Encoder memory   │
  │ V（取什么内容）  │ 本句自己          │ Encoder memory   │
  │ 作用            │ 理解句内关系      │ 对齐源句与译文    │
  └─────────────────┴──────────────────┴──────────────────┘

  公式（与 06.3 相同）：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

  本项目中 DecoderLayer 的调用（见 transformer.py）：
    cross_attn(query=x, key=memory, value=memory, mask=memory_mask)
    # x = Decoder 当前层输入（中文侧）
    # memory = Encoder 输出（英文侧）
""")

    # ------------------------------------------------------------------
    # Part 1: 手工小例子 ——「i love you」→ 写「我爱你」
    # ------------------------------------------------------------------
    en_words = ["i", "love", "you"]
    zh_steps = ["<bos>", "我", "爱", "你"]  # Decoder 已生成的部分（Teacher Forcing 视角）
    L_src, L_tgt, d = len(en_words), len(zh_steps), 6

    print("=" * 60)
    print("【Part 1】翻译对齐直觉：i love you → 我爱你")
    print("=" * 60)
    print(f"\n  英文（Encoder memory，提供 K 和 V）: {en_words}")
    print(f"  中文（Decoder 各位置，提供 Q）:         {zh_steps}")
    print("  下面用手工向量模拟「训练好后」的对齐模式（真实模型是学出来的）。")

    # 构造近似 one-hot 的向量，使「我→i」「爱→love」「你→you」对齐清晰
    memory = torch.zeros(L_src, d)
    for i in range(L_src):
        memory[i, i] = 3.0  # Encoder 输出（K、V 同源）

    queries = torch.zeros(L_tgt, d)
    queries[0, 0] = 1.0          # <bos> 略偏向句首
    queries[1, 0] = 3.0          # 我 → i
    queries[2, 1] = 3.0          # 爱 → love
    queries[3, 2] = 3.0          # 你 → you

    Q, K, V = queries, memory, memory
    out, weights, scores = scaled_dot_product_attention(Q, K, V)

    print(f"\n  Q shape: {tuple(Q.shape)}  （L_tgt={L_tgt}，每个中文位置一个查询）")
    print(f"  K shape: {tuple(K.shape)}  （L_src={L_src}，每个英文位置一个键）")
    print(f"  scores shape: {tuple(scores.shape)}  →  对齐矩阵 (中文行 × 英文列)")
    print(f"  output shape: {tuple(out.shape)}  →  每个中文位置融合原文后的向量")

    print_attention_matrix(zh_steps, en_words, weights, "【对齐权重】行=中文 Query，列=英文 Key")

    print("""
【读热力图】
  - 行「爱」若在列「love」权重最高 → 写「爱」时主要参考英文 love
  - 这就是「软对齐」：不是硬规则，而是模型学出的概率分布
  - 旧 Seq2Seq 用单一 context 向量；Transformer 让每个译文字都能查整句原文
""")

    # ------------------------------------------------------------------
    # Part 2: Decoder 内的位置 —— Cross-Attn 在 Masked Self-Attn 之后
    # ------------------------------------------------------------------
    print("=" * 60)
    print("【Part 2】Decoder 一层里的数据流")
    print("=" * 60)
    print("""
  中文 tgt_in -> Embedding + PE -> x
                                      |
                    +-----------------+-----------------+
                    v                                   |
            Masked Self-Attn                            |
            Q,K,V 都来自 x                              |
            （只能看已写出的中文 + Causal Mask）          |
                    |                                   |
                    v                                   |
            Cross-Attn  <---- memory（Encoder 输出）----+
            Q 来自上一步的 x
            K, V 来自 memory
                    |
                    v
                  FFN -> 下一层 / 输出 logits

  要点：
    - Self-Attn 先让中文内部互相看见（「我」和「爱」的语序关系）
    - Cross-Attn 再让每个中文位置去英文里「查资料」
    - 两层分工：句内连贯 + 源句对齐
""")

    # ------------------------------------------------------------------
    # Part 3: 多头 Cross-Attention + shape
    # ------------------------------------------------------------------
    print("=" * 60)
    print("【Part 3】项目里的 MultiHead Cross-Attention（shape 追踪）")
    print("=" * 60)

    B, d_model, n_heads = 1, 16, 4
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)

    # 模拟：英文 5 个 token（含 pad），中文 4 个 token
    L_src_demo, L_tgt_demo = 5, 4
    memory_batch = torch.randn(B, L_src_demo, d_model)   # Encoder 输出
    decoder_x = torch.randn(B, L_tgt_demo, d_model)      # Decoder 隐状态

    cross_out = mha(decoder_x, memory_batch, memory_batch, mask=None)

    print(f"\n  decoder_x (Query):  {tuple(decoder_x.shape)}   batch, L_tgt, d_model")
    print(f"  memory  (Key/Val):  {tuple(memory_batch.shape)}   batch, L_src, d_model")
    print(f"  cross_out:          {tuple(cross_out.shape)}   shape 与 Query 相同")
    print(f"  内部 scores:        (B={B}, n_heads={n_heads}, L_tgt={L_tgt_demo}, L_src={L_src_demo})")
    print("""
  Cross-Attn 与 Self-Attn 用的是同一个 MultiHeadAttention 类；
  区别仅在于 forward(query, key, value) 的三个参数是否同源。
""")

    # ------------------------------------------------------------------
    # Part 4: 源句 Pad Mask
    # ------------------------------------------------------------------
    print("=" * 60)
    print("【Part 4】源句 Pad Mask —— 英文补齐位不能被查")
    print("=" * 60)

    # ids: bos, i, love, you, eos, pad, pad
    src_ids = torch.tensor([[1, 4, 36, 8, 2, 0, 0]])
    pad_idx = 0
    src_mask = make_pad_mask(src_ids, pad_idx)  # (B, 1, 1, L_src)

    print(f"  src ids:  {src_ids[0].tolist()}")
    print(f"  有效位:   {src_mask[0, 0, 0].int().tolist()}  （1=有效, 0=pad）")
    print("""
  Cross-Attn 的 mask 作用在「英文列」上：
    - 写中文时，不能把注意力分配到英文的 <pad> 位置
    - 实现：scores 在 pad 列填 -inf，softmax 后权重为 0
  注意：Cross-Attn 不需要 Causal Mask（那是 Decoder Self-Attn 的事）
        写「你」时，可以 attend 到英文任意有效位置，包括未来的 you
""")

    # 演示 pad mask 效果（单头）
    L_src_pad = src_ids.size(1)
    Q_demo = torch.randn(L_tgt, d)
    K_demo = torch.randn(L_src_pad, d)
    V_demo = K_demo.clone()
    pad_mask_2d = src_mask[0, 0, 0].unsqueeze(0).expand(L_tgt, -1)  # broadcast 到 (L_tgt, L_src)
    _, weights_masked, _ = scaled_dot_product_attention(Q_demo, K_demo, V_demo, mask=pad_mask_2d)

    print("  pad 列（最后两列）的注意力权重之和应 ≈ 0：")
    pad_cols = weights_masked[:, -2:].sum(dim=1)
    for i, zh in enumerate(zh_steps):
        print(f"    位置「{zh}」对 pad 列权重和: {pad_cols[i].item():.6f}")

    # ------------------------------------------------------------------
    # Part 5: 与 RNN Seq2Seq 的对比
    # ------------------------------------------------------------------
    print("""
【Part 5】为什么 Cross-Attention 比旧 Seq2Seq 强？

  旧 Encoder-Decoder（RNN）:
    整句英文压成「一个」固定长度向量 → Decoder 只能反复读这一坨
    长句信息容易丢失，对齐是隐式的

  Transformer Cross-Attention:
    memory 保留每个英文位置的向量 → Decoder 每步可以「点名」查任意位置
    对齐矩阵 (L_tgt × L_src) 可解释、可可视化

【三种注意力分工（复习）】
  Encoder Self-Attn:   Q,K,V <- 英文        -> 理解原文内部关系
  Decoder Self-Attn:   Q,K,V <- 中文        -> 已生成译文的语序与连贯（+ Causal）
  Cross-Attn:          Q <- 中文, K,V <- memory -> 译文字 <-> 原文字对齐
""")

    # ------------------------------------------------------------------
    # 热力图
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(weights.detach().numpy(), cmap="Oranges", vmin=0, vmax=1)
    axes[0].set_xticks(range(L_src))
    axes[0].set_yticks(range(L_tgt))
    axes[0].set_xticklabels(en_words)
    axes[0].set_yticklabels(zh_steps)
    axes[0].set_xlabel("Key（英文 / memory）")
    axes[0].set_ylabel("Query（中文 / Decoder）")
    axes[0].set_title("Cross-Attention: i love you → 我爱你")
    for i in range(L_tgt):
        for j in range(L_src):
            axes[0].text(j, i, f"{weights[i, j].item():.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # 架构示意图（文字）
    axes[1].axis("off")
    diagram = """
Encoder (英文)                Decoder (中文)
-------------                -------------
i   -> h_i  --+
love-> h_love-+-- memory --> Cross-Attn <-- Q(我/爱/你...)
you -> h_you --+              ^
                              Masked Self-Attn
                              （只看已写中文）

scores[i,j] = 中文位置 i 对英文位置 j 的关注度
"""
    axes[1].text(0.05, 0.5, diagram, fontsize=11, va="center")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "06_cross_attention_heatmap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"热力图已保存: {path}")
    print("\n下一步: python 07_model_assembly.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
