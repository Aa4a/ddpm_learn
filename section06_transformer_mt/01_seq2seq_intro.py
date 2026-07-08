# -*- coding: utf-8 -*-
"""
Section 06.1 - 机器翻译在做什么？（Seq2Seq 直觉）

本脚本不涉及神经网络，只建立「输入一句英文 → 输出一句中文」的整体图景。
建议作为 Section 06 的第一站。
"""

from data_utils import BUILTIN_PAIRS


def main():
    print("=" * 60)
    print("Section 06.1 - 机器翻译（Seq2Seq）直觉")
    print("=" * 60)

    print("""
【问题】
  输入：一句英文  "i love you"
  输出：一句中文  "我爱你"

【为什么不能直接「查字典逐词替换」？】
  英文和中文的语序、词数往往不对齐：
    good morning  →  早上好   （2 个英文词 → 3 个汉字）
    where is the book  →  书在哪里  （语序也变了）

  所以需要模型理解「整句意思」，再生成「整句译文」。
  这类「序列 → 序列」任务叫 Seq2Seq（Sequence to Sequence）。
""")

    print("【Encoder-Decoder 两阶段（后面小节会逐步实现）】")
    print("""
  英文句子 ──► Encoder（编码器）──► 句向量 memory（整句的压缩表示）
                                           │
  中文句子 ◄── Decoder（解码器）◄──────────┘
              （一个字一个字往外写）

  训练时：我们同时有英文和中文，让模型学「看到英文后，该写哪个中文」。
  推理时：只有英文，Decoder 从 <bos> 开始，逐个预测汉字，直到 <eos>。
""")

    print("【本仓库内置的平行语料（前 5 对）】")
    for i, (en, zh) in enumerate(BUILTIN_PAIRS[:5], 1):
        print(f"  {i}. EN: {en}")
        print(f"     ZH: {zh}")

    print(f"\n  … 共 {len(BUILTIN_PAIRS)} 对，无需联网下载。")

    print("""
【接下来学什么？】（按顺序，不要跳）
  06.2  02_data_vocab.py      文字怎么变成数字（分词、词表）
  06.3  03_attention_intuition.py  注意力：翻译时「看哪里」
  06.4  04_positional_encoding.py  位置编码：词序信息
  06.5  05_multihead_and_mask.py   多头注意力 + Mask
  06.6  06_model_assembly.py       拼成完整 Transformer
  06.7  07_train.py                训练
  06.8  08_infer.py                推理翻译
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
