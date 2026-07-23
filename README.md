# Denoising Diffusion Probabilistic Models (DDPM) 学习笔记

> 参考论文：Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

## 环境

- Conda 环境名：`ddpm_learn`
- Python 3.10 | PyTorch 2.12 | NumPy | Matplotlib | Jupyter

```bash
conda activate ddpm_learn
```

## 课程结构

| 文件夹 | 内容 | 状态 |
|--------|------|------|
| `section01_intro/` | 直觉概览、正向加噪公式、Beta Schedule 可视化 | ✅ 完成 |
| `hf_diffusion/` | 对照 `hf_diffusion_train.py` 的数学详解 + 纯 NumPy 可视化 | ✅ 完成 |
| `section02_forward_math/` | 正向过程完整数学推导、马尔可夫链 | 🔜 |
| `section02_reverse_process/` | 反向过程后验推导（贝叶斯+配方法）、蒙特卡洛验证、反向采样 | ✅ 完成 |
| `section03_loss/` | ELBO → 简化损失函数（预测噪声 ε） | ✅ 完成 |
| `section04_unet/` | 时间嵌入 + U-Net 架构实现 | ✅ 完成 |
| `section05_train_sample/` | 完整训练循环 + 采样（MNIST） | ✅ 完成 |
| `section06_transformer_mt/` | Transformer 机器翻译（10 个小节：Seq2Seq → 注意力 → 训练 → 推理） | ✅ 完成 |
| `section07_stable_diffusion/` | 迷你 Stable Diffusion（VAE + 文本 Cross-Attn + CFG，MNIST 文生图） | ✅ 完成 |
| `section08_masked_diffusion/` | 离散 Mask 扩散（通往 LLaDA：双向填空 + 迭代 Unmask） | ✅ 完成 |
| `section09_torcheeg_diffusion/` | TorchEEG 扩散（BUNet/BCUNet + DDPMTrainer，架构与原理） | ✅ 完成 |

## 快速入口

- **Section 01 理论**：[`section01_intro/README.md`](section01_intro/README.md)
- **HF 训练脚本数学对照**（无需 diffusers/GPU）：[`hf_diffusion/README.md`](hf_diffusion/README.md)
- **训练脚本本体**（需 GPU + diffusers）：[`hf_diffusion_train.py`](hf_diffusion_train.py)
- **Section 04 U-Net 与时间嵌入**：[`section04_unet/README.md`](section04_unet/README.md)
- **Section 05 训练与采样**：[`section05_train_sample/README.md`](section05_train_sample/README.md)
- **Section 06 Transformer 翻译**：[`section06_transformer_mt/README.md`](section06_transformer_mt/README.md)
- **Section 07 迷你 Stable Diffusion**：[`section07_stable_diffusion/README.md`](section07_stable_diffusion/README.md)
- **Section 08 Mask 扩散 → LLaDA**：[`section08_masked_diffusion/README.md`](section08_masked_diffusion/README.md)
- **Section 09 TorchEEG 扩散**：[`section09_torcheeg_diffusion/README.md`](section09_torcheeg_diffusion/README.md)

## 核心公式速查

**正向过程（闭合采样）**
$$x_t = \sqrt{\bar\alpha_t}\cdot x_0 + \sqrt{1-\bar\alpha_t}\cdot\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)$$

**Beta Schedule**
$$\beta_t \in [10^{-4},\ 0.02],\quad \alpha_t = 1-\beta_t,\quad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s$$
