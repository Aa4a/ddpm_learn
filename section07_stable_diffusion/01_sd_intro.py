# -*- coding: utf-8 -*-
"""
Section 07.1 - Stable Diffusion 在做什么？（直觉）

本脚本不训练网络，只建立整体图景：
  像素 DDPM -> 潜空间扩散（LDM）-> 文本条件 + CFG
建议作为 Section 07 的第一站。
"""

from data_utils import setup_stdio


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 07.1 - Stable Diffusion（迷你 Latent Diffusion）直觉")
    print("=" * 60)

    print("""
【回顾】Section 01~05 的像素 DDPM
  直接在图像像素 x 上加噪 / 去噪：
    x_T ~ N(0,I)  --1000 步-->  x_0（生成图）

  问题：分辨率一高（256x256、512x512），U-Net 算力爆炸。

【Stable Diffusion 的核心想法 = Latent Diffusion】
  不在像素上扩散，而在「压缩后的潜变量 z」上扩散：

    真实图 x  --VAE Encoder-->  z0（小很多）
    z_T ~ N(0,I) --U-Net 去噪--> z0
    z0 --VAE Decoder--> 生成图 x_hat

  潜空间空间尺寸小（本教学版：28x28 -> 7x7，通道 4），
  扩散网络便宜一个数量级，才能做高分辨率文生图。
""")

    print("【SD 三大件】（真实 SD 与本教学版一一对应）")
    print("""
  +------------------+------------------------+--------------------+
  | 组件             | 真实 Stable Diffusion  | 本仓库迷你版        |
  +------------------+------------------------+--------------------+
  | VAE              | KL-VAE（约 8x 下采样） | TinyVAE（约 4x）    |
  | Text Encoder     | CLIP / OpenCLIP        | 数字名词嵌入        |
  | Cond U-Net       | Cross-Attn U-Net       | CondUNet（7x7）     |
  | 条件增强         | Classifier-Free Guid.  | 同左（CFG）         |
  +------------------+------------------------+--------------------+
""")

    print("【文本怎么控制图像？—— Cross-Attention】")
    print("""
  你在 Section 06.6 已经学过：翻译时 Decoder(Q) 查 Encoder(K/V)。

  文生图里完全同构：
    Q <- 图像（潜空间）特征
    K,V <- 文本 embedding
  每个空间位置问：「文字里哪个词和我现在画的内容最相关？」

  再加 Classifier-Free Guidance（CFG）：
    eps = eps_uncond + s * (eps_cond - eps_uncond)
  s 越大，越「听话」地跟 prompt，但可能过饱和。
""")

    print("【本仓库的教学任务】")
    print("""
  输入 prompt：  "three"
  输出图像：     一张手写数字 3（MNIST 风格）

  目的是跑通 SD 全流程，不是工业画质。
  真实 SD 换更大的 VAE / CLIP / U-Net + 海量图文对即可。
""")

    print("""
【接下来学什么？】（按顺序）
  07.2  02_vae_latent.py         潜空间：压缩与重建
  07.3  03_text_encoder.py       文本编码（简化 CLIP）
  07.4  04_cross_attn_unet.py    U-Net 里的 Cross-Attention
  07.5  05_cfg.py                Classifier-Free Guidance
  07.6  06_model_assembly.py     组装完整 LDM + 维度检查
  07.7  07_train.py              训练 VAE -> 潜空间扩散
  07.8  08_sample.py             按 prompt 采样
  07.9  09_visualize_cross_attn.py  看模型「看了哪个词」
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
