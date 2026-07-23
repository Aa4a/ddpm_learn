# -*- coding: utf-8 -*-
"""
Section 08.3 - 双向注意力 vs 因果掩码

自回归 LM：预测位置 i 时不能看 i 右侧（Causal Mask）。
Mask 扩散：被抹掉的位置需要左右上下文一起猜 —— 必须双向。
"""

from data_utils import setup_stdio


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 08.3 - 双向 vs 因果（为何 Mask 扩散不用 Causal）")
    print("=" * 60)

    print("""
【场景】句子部分被抹掉：

  <bos> i <mask> you <eos>

要猜 <mask> 是什么，人会同时看左边的 i 和右边的 you → 「love」很合理。
如果强制 Causal（只能看左边）：只看到 「i」—— 信息不够。
""")

    print("【注意力可见性】")
    print("""
  自回归 Decoder（§06）因果掩码 —— 下三角：

           i    love   you
    i      ✓     ✗      ✗
    love   ✓     ✓      ✗
    you    ✓     ✓      ✓

  Mask 扩散 / LLaDA 双向 —— 全可见（只挡 pad）：

           i   <mask>  you
    i      ✓     ✓      ✓
    <mask> ✓     ✓      ✓
    you    ✓     ✓      ✓

  因此骨干是 Encoder-only Transformer，而不是带 Causal 的 Decoder。
""")

    print("【和 BERT 的关系（容易混淆）】")
    print("""
  BERT MLM:
    - 也是双向 + 预测 mask
    - 掩码率通常固定 ~15%
    - 目标偏「表示学习」，不是按扩散过程采样生成长文

  Masked Diffusion / LLaDA:
    - 掩码率 t ~ Uniform(0,1)（从几乎不遮到几乎全遮）
    - 训练目标可从离散扩散似然下界推出来
    - 推理：从全 mask 迭代 Unmask，是真正的生成模型

  一句话：BERT 像「完形填空练习」；LLaDA 像「从空白稿反复改到成文」。
""")

    print("【条件生成怎么做？（预习 LLaDA SFT）】")
    print("""
  不必上 Cross-Attn：
    Prompt 位置永远不 mask、采样时始终可见；
    只 mask / 只生成 Response 部分。

  例：  [User 问题........] [ <mask> <mask> ... <mask> ]
         ↑ 固定可见              ↑ 扩散填空
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
