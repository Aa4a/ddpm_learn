# -*- coding: utf-8 -*-
"""
Section 06.2 - 从文字到数字：分词、词表、Batch

神经网络只能处理数字。本节演示：
  1. 英文按空格分词，中文按字分词
  2. 特殊符号 <pad> <bos> <eos> <unk>
  3. encode / decode 往返
  4. 一个 mini-batch 长什么样
"""

from data_utils import (
    BOS,
    EOS,
    PAD,
    build_vocab,
    load_builtin_pairs,
    make_dataloader,
    tokenize_en,
    tokenize_zh,
    TranslationDataset,
)


def main():
    print("=" * 60)
    print("Section 06.2 - 分词与词表")
    print("=" * 60)

    # --- 1. 分词 ---
    en = "i love you"
    zh = "我爱你"
    en_toks = tokenize_en(en)
    zh_toks = tokenize_zh(zh)
    print("\n【1】分词")
    print(f"  英文 '{en}'  →  {en_toks}   （按空格切）")
    print(f"  中文 '{zh}'  →  {zh_toks}   （按字切）")

    # --- 2. 词表 ---
    pairs = load_builtin_pairs()
    src_vocab = build_vocab([tokenize_en(p.src) for p in pairs])
    tgt_vocab = build_vocab([tokenize_zh(p.tgt) for p in pairs])

    print("\n【2】词表（字典：词/字 → 整数 id）")
    print(f"  源语言词表大小: {len(src_vocab)}  （含 {PAD}, {BOS}, {EOS}, <unk>）")
    print(f"  目标语言词表大小: {len(tgt_vocab)}")
    print(f"  例: 'love' → id {src_vocab.token_to_id.get('love', src_vocab.unk_idx)}")
    print(f"      '爱'   → id {tgt_vocab.token_to_id.get('爱', tgt_vocab.unk_idx)}")

    # --- 3. encode / decode ---
    src_ids = src_vocab.encode(en_toks, add_bos_eos=True)
    tgt_ids = tgt_vocab.encode(zh_toks, add_bos_eos=True)

    print("\n【3】编码：tokens → id 列表")
    print(f"  英文 ids: {src_ids}")
    print(f"    含义:   [{BOS}] + {en_toks} + [{EOS}]")
    print(f"  中文 ids: {tgt_ids}")
    print(f"    含义:   [{BOS}] + {zh_toks} + [{EOS}]")

    print("\n【4】解码：ids → 文字（验证可逆）")
    print(f"  还原英文: {src_vocab.decode(src_ids)}")
    print(f"  还原中文: {tgt_vocab.decode(tgt_ids)}")

    # --- 4. batch ---
    ds = TranslationDataset(pairs[:4], src_vocab, tgt_vocab)
    batch = next(iter(make_dataloader(ds, batch_size=2, shuffle=False)))

    print("\n【5】一个 Batch（2 句话拼在一起，短句末尾补 pad）")
    print(f"  src shape: {batch['src'].shape}   # (batch_size, 最大句长)")
    print(f"  src[0]:    {batch['src'][0].tolist()}")
    print(f"  src[1]:    {batch['src'][1].tolist()}  ← 短的句子后面是 pad(0)")
    print(f"  对应英文:  {batch['src_text']}")

    print("""
【小结】
  - 模型输入的不是字符串，而是整数矩阵 src / tgt
  - <bos> 表示「开始写」，<eos> 表示「写完了」，<pad> 只是对齐长度用
  - 训练时 pad 位置的 loss 会被忽略（后面 07_train.py）

下一步: python 03_attention_intuition.py
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
