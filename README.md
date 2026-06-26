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
| `section04_unet/` | 时间嵌入 + U-Net 架构实现 | 🔜 |
| `section05_train_sample/` | 完整训练循环 + 采样（MNIST） | 🔜 |

## 快速入口

- **Section 01 理论**：[`section01_intro/README.md`](section01_intro/README.md)
- **HF 训练脚本数学对照**（无需 diffusers/GPU）：[`hf_diffusion/README.md`](hf_diffusion/README.md)
- **训练脚本本体**（需 GPU + diffusers）：[`hf_diffusion_train.py`](hf_diffusion_train.py)
- **Section 03 损失推导**：[`section03_loss/README.md`](section03_loss/README.md)

## 核心公式速查

**正向过程（闭合采样）**
$$x_t = \sqrt{\bar\alpha_t}\cdot x_0 + \sqrt{1-\bar\alpha_t}\cdot\varepsilon, \quad \varepsilon\sim\mathcal{N}(0,I)$$

**Beta Schedule**
$$\beta_t \in [10^{-4},\ 0.02],\quad \alpha_t = 1-\beta_t,\quad \bar\alpha_t = \prod_{s=1}^{t}\alpha_s$$
