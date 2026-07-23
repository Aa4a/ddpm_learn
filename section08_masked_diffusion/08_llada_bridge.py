# -*- coding: utf-8 -*-
"""
Section 08.8 - 通向 LLaDA：概念桥接

总结本节与自回归 / BERT / LLaDA 的关系，给出阅读顺序。
"""

from data_utils import setup_stdio


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 08.8 - 通向 LLaDA")
    print("=" * 60)

    print("""
你已经在本仓库跑通的迷你版 Masked Diffusion，对应 LLaDA 的骨架：

  正向：随机 mask（t ~ U）
  模型：双向 Transformer 预测 mask 位
  反向：多步 Unmask（可 remask）
  条件：prompt 保持可见（SFT 同构）
""")

    print("【三方对照】")
    print("""
  +----------------+------------------+------------------+--------------------+
  |                | 自回归 LM (§06)  | BERT MLM         | LLaDA / 本节        |
  +----------------+------------------+------------------+--------------------+
  | 注意力         | Causal           | Bidirectional    | Bidirectional       |
  | 训练掩码       | 无（预测下一词） | 固定 ~15%        | t ~ U(0,1)          |
  | 似然含义       | 精确因子分解     | 非生成导向       | 扩散 NELBO / 下界   |
  | 采样           | 左→右逐 token    | （一般不生成）   | 并行迭代 Unmask     |
  | 典型优势       | 成熟、生态好     | 表示 / NLU       | 双向、缓解 reversal |
  +----------------+------------------+------------------+--------------------+
""")

    print("【读 LLaDA 论文时可以对号入座】")
    print("""
  论文里你会看到的说法          ↔  本节对应
  ---------------------------------------------------------
  absorbing discrete diffusion  ↔  <mask> 作为吸收噪声态
  mask predictor Transformer    ↔  MaskPredictor（Encoder-only）
  pretraining: random mask t    ↔  06_train + q_sample
  SFT: 只 mask response         ↔  07_sample --prompt ...
  low-confidence remasking      ↔  sample() 里按置信度揭开
  reversal curse 更弱           ↔  双向可见左右文（直觉）
""")

    print("【建议阅读顺序】")
    print("""
  1. 本节脚本按 01→07 跑通（务必 --fast 训一次再 sample）
  2. （可选）D3PM / Absorbing diffusion / MDLM 短文 — 补离散扩散公式
  3. LLaDA: Large Language Diffusion Models (Nie et al., 2025)
     https://arxiv.org/abs/2502.09992
     项目页: https://ml-gsai.github.io/LLaDA-demo/

  读论文时重点看：
    - 前向 mask 过程与训练目标如何写成似然下界
    - 预训练 vs SFT 的 mask 范围差异
    - 采样步数、remask 策略与规模实验结论
""")

    print("【本节刻意没做的】")
    print("""
  - 亿级参数 / 万亿 token（教学语料只有几十句）
  - 完整离散扩散转移矩阵推导（需要可另开数学专节）
  - 与真实 LLaDA 权重对齐

  目标是：概念对齐后，读论文不再「完全陌生」。
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
