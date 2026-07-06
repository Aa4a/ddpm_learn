# Section 05 - 完整训练循环与采样（MNIST）

## 本节目标

- 将 Section 01~04 的知识串联：**Schedule → 加噪 → 预测噪声 → 反向采样**
- 实现完整的 DDPM 训练循环（简化损失 $L_{\rm simple}$）
- 实现从纯高斯噪声 $x_T$ 出发的 **1000 步反向去噪采样**
- 在 MNIST 上训练简易 U-Net，亲眼见证模型「凭空」生成手写数字

## 本节目录

- 零、从公式到代码：完整流程一览
- 一、训练循环
- 二、反向采样
- 三、文件说明与运行方式
- 四、预期效果与调参建议

---

## 零、从公式到代码：完整流程一览

```
┌─────────────────────────────────────────────────────────────┐
│                        训练阶段                              │
├─────────────────────────────────────────────────────────────┤
│  x₀ ~ 数据分布（MNIST 真实图片）                              │
│  t  ~ Uniform{0, ..., T-1}                                  │
│  ε  ~ N(0, I)                                               │
│  x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε        ← 正向加噪（Section 01）│
│  ε_θ = UNet(x_t, t)                  ← 网络预测（Section 04）│
│  L = ||ε - ε_θ||²                    ← 简化损失（Section 03）│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                        采样阶段                              │
├─────────────────────────────────────────────────────────────┤
│  x_T ~ N(0, I)                       ← 从纯噪声出发          │
│  for t = T-1, ..., 0:                                       │
│    ε_θ = UNet(x_t, t)                                       │
│    μ_θ = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ)  ← Section 02     │
│    x_{t-1} = μ_θ + √β̃_t z   (t>0),  z~N(0,I)              │
│    x_0 = μ_θ                  (t=0)                         │
└─────────────────────────────────────────────────────────────┘
```

| 章节 | 本节中的对应代码 |
|------|-----------------|
| Section 01 | `DDPMSchedule.q_sample()` — 正向加噪闭合公式 |
| Section 02 | `DDPMSchedule.p_sample()` — 噪声形式的 $\tilde\mu_t$ |
| Section 03 | `DDPMSchedule.training_loss()` — MSE($\varepsilon$, $\varepsilon_\theta$) |
| Section 04 | `load_simple_unet()` — 加载 Section 04 的 SimpleUNet |

---

## 一、训练循环

### 1.1 核心代码

每一步训练的逻辑（见 `ddpm.py`）：

```python
def training_loss(self, model, x0):
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt, _ = self.q_sample(x0, t, noise=noise)   # 正向加噪
    noise_pred = model(xt, t.float())            # 网络预测噪声
    return F.mse_loss(noise_pred, noise)          # 简化损失
```

与 HuggingFace `hf_diffusion_train.py` 的训练逻辑完全一致，只是我们使用自己实现的 SimpleUNet，无需 diffusers / accelerate。

### 1.2 数据预处理

MNIST 原始像素 $[0, 255]$ → `ToTensor()` → $[0, 1]$ → `Normalize(0.5, 0.5)` → **$[-1, 1]$**。

这与 DDPM 论文及 diffusers 的惯例一致：数据与噪声在同一尺度上。

### 1.3 优化器

- **AdamW**，学习率 $10^{-4}$
- 默认 batch size = 128，epochs = 20

---

## 二、反向采样

### 2.1 单步去噪公式

给定当前 $x_t$ 和网络预测的 $\varepsilon_\theta(x_t, t)$，后验均值的噪声形式（Section 02）：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right)$$

采样更新：

$$x_{t-1} = \mu_\theta + \sqrt{\tilde\beta_t}\,z, \quad z \sim \mathcal{N}(0, I) \quad (t > 0)$$

$$x_0 = \mu_\theta \quad (t = 0, \text{不再加噪})$$

### 2.2 采样循环

```python
@torch.no_grad()
def sample(self, model, shape, device):
    x = torch.randn(shape, device=device)          # x_T ~ N(0,I)
    for t in reversed(range(T)):                   # T-1 → 0
        x = self.p_sample(model, x, t)
    return x
```

从 $x_T$（纯噪声）出发，执行 1000 步反向去噪，最终得到 $x_0$（生成的手写数字）。

---

## 三、文件说明与运行方式

| 文件 | 说明 |
|------|------|
| `ddpm.py` | Beta Schedule、正向加噪、训练损失、反向采样 |
| `01_train_mnist.py` | MNIST 训练主脚本，定期采样 + 保存 checkpoint |
| `02_sample.py` | 从 checkpoint 独立采样 |
| `checkpoints/` | 模型权重（`.pth`，已 gitignore） |
| `figures/` | 采样图与损失曲线 |
| `data/` | MNIST 自动下载目录（已 gitignore） |

### 环境

```bash
conda activate ddpm_learn
```

依赖：`torch`, `torchvision`, `matplotlib`, `tqdm`（均已在项目环境中）。

### 完整训练（推荐 GPU）

```bash
cd section05_train_sample
python 01_train_mnist.py --epochs 20
```

### 快速演示（CPU 也可，约数分钟）

```bash
python 01_train_mnist.py --fast
```

`--fast` 模式：2 epoch + 5000 样本子集，每个 epoch 采样一次，用于验证流程是否正常。

### 从 checkpoint 采样

```bash
python 02_sample.py
# 或指定 checkpoint
python 02_sample.py --checkpoint checkpoints/unet_epoch_020.pth --num-samples 64
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 128 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--sample-every` | 5 | 每 N epoch 采样 |
| `--num-samples` | 64 | 每次采样 8×8 网格 |
| `--fast` | - | 快速演示模式 |
| `--device` | 自动 | `cuda` 或 `cpu` |

---

## 四、预期效果与调参建议

### 4.1 训练损失

- 初始 loss 约 0.5~1.0（随机预测噪声）
- 随训练下降，20 epoch 后通常可降至 0.05~0.15

### 4.2 采样质量

| Epoch | 预期现象 |
|-------|----------|
| 1~2 | 模糊色块，隐约可见数字轮廓 |
| 5~10 | 可辨认的数字形状，细节模糊 |
| 20+ | 较清晰的手写数字，风格多样 |

> MNIST 仅 28×28 灰度图，配合 Section 04 的简易 U-Net（2 层下采样），20 epoch 已可生成可辨认的数字。追求更高质量可增加 epoch 或换更大 U-Net。

### 4.3 调参建议

- **loss 不下降**：检查数据归一化是否为 $[-1, 1]$，学习率是否过大
- **采样全噪声**：训练不充分，或 checkpoint 路径错误
- **CPU 太慢**：使用 `--fast` 验证流程，正式训练建议 GPU + `--epochs 30`

---

## 与 hf_diffusion_train.py 的对比

| | 本节 `01_train_mnist.py` | `hf_diffusion_train.py` |
|--|--------------------------|-------------------------|
| 数据集 | MNIST (28×28, 1ch) | Butterflies (128×128, 3ch) |
| 网络 | Section 04 SimpleUNet | diffusers UNet2DModel |
| 依赖 | 仅 PyTorch | diffusers + accelerate + datasets |
| 目的 | 理解完整流程 | 生产级训练参考 |

---

## 课程总结

恭喜你完成了 DDPM 从零到一的完整学习路径：

| 章节 | 内容 |
|------|------|
| Section 01 | 正向加噪直觉与 Beta Schedule |
| Section 02 | 反向后验推导与采样公式 |
| Section 03 | ELBO → 简化噪声预测损失 |
| Section 04 | 时间嵌入 + U-Net 架构 |
| **Section 05** | **训练 + 采样，亲手生成图片** |
