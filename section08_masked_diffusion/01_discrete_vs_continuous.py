# -*- coding: utf-8 -*-
"""
Section 08.1 - 连续扩散 vs 离散 Mask 扩散（直觉）

本脚本不训练网络，只建立：为什么文本不能直接套用 DDPM 的高斯加噪，
以及 LLaDA 类方法如何用 <mask> 代替噪声。
"""

from data_utils import setup_stdio


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 08.1 - 连续扩散 vs 离散 Mask 扩散")
    print("=" * 60)

    print("""
【回顾】你已经会的：连续 DDPM（图像）
  正向：x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε ,  ε ~ N(0,I)
  反向：网络预测 ε，逐步去噪得到图像

  「半个像素」有意义 —— 灰度可以连续插值。

【问题】文本为什么不能直接加高斯噪声？
  token 是离散的：词表上的整数 id
  「love 和 hate 的中间值」没有语义
  把 embedding 加噪再取最近邻，往往变成胡言乱语，且难有干净的马尔可夫结构

【离散扩散的一种答案：Masked Diffusion】
  把「噪声」定义成特殊符号 <mask>（吸收态）：

    连续:  清晰图像 ──加噪──► 纯噪声
    离散:  清晰句子 ──随机抹词──► 几乎全是 <mask>

  反向不是预测 ε，而是：
    看部分可见词 + 双向上下文 → 预测被抹掉的原 token
""")

    print("【一张对照表】")
    print("""
  +------------------+---------------------------+-----------------------------+
  |                  | 连续 DDPM (§01-05)        | Mask 扩散 (本节 / LLaDA)    |
  +------------------+---------------------------+-----------------------------+
  | 数据             | 像素 / 潜变量 z           | token 序列                  |
  | 「噪声」         | 高斯 ε                    | <mask> 替换                 |
  | 时间 t           | 噪声强度 ᾱ_t              | 掩码比例 t ∈ (0,1]          |
  | 网络预测         | ε 或 x0（连续）           | 被 mask 位置的原 token      |
  | 损失             | MSE                       | CrossEntropy（mask 位）     |
  | 骨干             | U-Net                     | 双向 Transformer            |
  | 采样             | 逐步去噪                  | 逐步 Unmask（可 remask）    |
  | 生成顺序         | 空间并行                  | 位置并行（非自回归）        |
  +------------------+---------------------------+-----------------------------+
""")

    print("【和你已学内容的衔接】")
    print("""
  §06 Transformer 翻译
    - 你有 Causal Decoder → 自回归逐词生成
    - Encoder 已是双向注意力
    - 本节：用「只有 Encoder」做生成（无 Causal Mask）

  §07 Stable Diffusion
    - 文本条件 + Cross-Attn 是图像侧技巧
    - LLaDA 本体不依赖 U-Net/VAE；条件生成靠「prompt 不 mask」即可

  本节目标：跑通「Mask 正向 → 双向预测 → 迭代 Unmask」迷你管线，
  为阅读 LLaDA 论文铺路（教学规模，不是 8B）。
""")

    print("""
【接下来学什么？】（按顺序）
  08.2  02_forward_masking.py       正向：随机 <mask>
  08.3  03_bidirectional_vs_causal.py  为什么不能用因果掩码
  08.4  04_training_objective.py    损失：mask 位 CE + 1/t
  08.5  05_model_assembly.py        组装 MaskPredictor
  08.6  06_train.py                 在短句上训练
  08.7  07_sample.py                迭代 Unmask 采样
  08.8  08_llada_bridge.py          对照 LLaDA / 下一步读什么
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
