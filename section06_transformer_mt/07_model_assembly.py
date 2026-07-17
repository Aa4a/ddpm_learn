# -*- coding: utf-8 -*-
"""
Section 06.7 - 组装完整 Transformer，检查维度

把前面学过的组件拼成 Encoder-Decoder，跑一次前向传播，确认 shape 正确。
"""

import torch

from data_utils import build_vocab, load_builtin_pairs, tokenize_en, tokenize_zh
from transformer import TransformerConfig, TransformerMT, make_pad_mask, make_tgt_mask


def main():
    print("=" * 60)
    print("Section 06.7 - Transformer 组装与维度检查")
    print("=" * 60)

    pairs = load_builtin_pairs()[:20]
    src_vocab = build_vocab([tokenize_en(p.src) for p in pairs])
    tgt_vocab = build_vocab([tokenize_zh(p.tgt) for p in pairs])

    cfg = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
        pad_idx=src_vocab.pad_idx,
    )
    model = TransformerMT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}  （教学用小模型）")

    # 构造一条样本
    en, zh = "i love you", "我爱你"
    src_ids = src_vocab.encode(tokenize_en(en), add_bos_eos=True)
    tgt_ids = tgt_vocab.encode(tokenize_zh(zh), add_bos_eos=True)

    src = torch.tensor([src_ids])
    tgt_full = torch.tensor([tgt_ids])
    tgt_in = tgt_full[:, :-1]   # Teacher Forcing 输入
    tgt_out = tgt_full[:, 1:]   # 要预测的标签

    src_mask = make_pad_mask(src, src_vocab.pad_idx)
    tgt_mask = make_tgt_mask(tgt_in, src_vocab.pad_idx)

    print("\n【数据】")
    print(f"  英文: {en}  →  src ids: {src_ids}")
    print(f"  中文: {zh}  →  tgt ids: {tgt_ids}")

    print("\n【Teacher Forcing 错位一行】")
    print(f"  tgt_in  (Decoder 输入): {tgt_in[0].tolist()}")
    print(f"  tgt_out (要预测的目标): {tgt_out[0].tolist()}")
    print("  每个位置：给定左边，预测下一个 token。")

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        dec_out = model.decode(tgt_in, memory, tgt_mask, src_mask)
        logits = model(src, tgt_in, src_mask, tgt_mask)

    print("\n【维度追踪】")
    print(f"  src:        {tuple(src.shape)}")
    print(f"  tgt_in:     {tuple(tgt_in.shape)}")
    print(f"  memory:     {tuple(memory.shape)}   ← Encoder 输出，给 Decoder 用")
    print(f"  dec_out:    {tuple(dec_out.shape)}")
    print(f"  logits:     {tuple(logits.shape)}   ← 每个位置对词表各类别的得分")
    print(f"  词表大小:   {len(tgt_vocab)}")

    print("""
【整体数据流】
  src ──► Embedding + PE ──► Encoder × N ──► memory
  tgt_in ──► Embedding + PE ──► Decoder × N ──► Linear ──► logits
                                    ↑
                                 cross-attn 读 memory

【接下来】
  08_train.py  用很多句子反复训练，让 logits 接近 tgt_out
  09_infer.py  没有 tgt_out 了，用 greedy_decode 一个字一个字生成

建议: python 08_train.py --fast
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
