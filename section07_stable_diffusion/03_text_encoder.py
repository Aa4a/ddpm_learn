# -*- coding: utf-8 -*-
"""
Section 07.3 - Text Encoder（简化版 CLIP）

演示：
  1. 数字名词表与 encode_prompt
  2. TextEncoder 前向：ids → (B, L, ctx_dim)
  3. <uncond> 用于 CFG 的无条件分支

运行：
  python 03_text_encoder.py
"""

from pathlib import Path

import torch

from data_utils import (
    DIGIT_NAMES,
    UNCOND_TOKEN,
    VOCAB,
    encode_prompt,
    decode_ids,
    uncond_ids,
    setup_stdio,
)
from sd_model import TextEncoder

HERE = Path(__file__).resolve().parent


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 07.3 - Text Encoder")
    print("=" * 60)

    print("""
【真实 SD】用 CLIP Text Encoder：把自然语言变成 token 序列，再编成向量。
【本教学版】词表极小，只有数字名 + 特殊符号——但接口相同：

  prompt 字符串  ->  token ids  ->  TextEncoder  ->  context (B, L, D)
""")

    print("【词表】")
    print(f"  {VOCAB}")
    print(f"  数字名: {DIGIT_NAMES}")
    print(f"  无条件: {UNCOND_TOKEN!r}（训练时随机替换；采样时 CFG 用）")

    prompts = ["three", "seven", "", "one two"]  # 多词会被截断到 MAX_TEXT_LEN=1
    print("\n【encode_prompt 示例】")
    for p in prompts:
        ids = encode_prompt(p)
        print(f"  {p!r:12s} -> ids={ids} -> decode={decode_ids(ids)!r}")

    enc = TextEncoder(ctx_dim=64)
    from data_utils import MAX_TEXT_LEN, UNCOND_ID
    rows = []
    for p in ["zero", "five", ""]:
        ids = encode_prompt(p)
        ids = ids + [UNCOND_ID] * (MAX_TEXT_LEN - len(ids))
        rows.append(ids[:MAX_TEXT_LEN])
    batch_ids = torch.tensor(rows, dtype=torch.long)

    ctx = enc(batch_ids)
    print(f"\n【TextEncoder 前向】")
    print(f"  text_ids shape : {tuple(batch_ids.shape)}")
    print(f"  context shape  : {tuple(ctx.shape)}  # (B, L, ctx_dim)")
    print(f"  uncond_ids     : {uncond_ids(2).tolist()}")

    print("""
【与 CLIP 的对应关系】

  +--------------+---------------------+------------------+
  |              | CLIP (真实 SD)      | 本教学版         |
  +--------------+---------------------+------------------+
  | 分词         | BPE / 77 token      | 空格切 + 数字名  |
  | 编码         | Transformer         | 浅层 Transformer |
  | 输出         | (B, 77, 768)        | (B, 1, 64)       |
  | 无条件       | 空串 / 学到的 null  | <uncond> token   |
  +--------------+---------------------+------------------+

【本节小结】
  - Text Encoder 把文字变成 U-Net Cross-Attn 可用的 K/V
  - 下一节：这些 context 如何注入去噪网络？
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
