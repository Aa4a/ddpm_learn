# Section 07 - 迷你 Stable Diffusion（Latent Diffusion）

> 参考：Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022（Stable Diffusion 的学术原型）

## 本节目标

- 理解 **为什么** 要从像素 DDPM 走到 **潜空间扩散（LDM）**
- 掌握 SD 三大件：**VAE / Text Encoder / 条件 U-Net（Cross-Attn）**
- 掌握 **Classifier-Free Guidance（CFG）**
- 在 MNIST 上跑通「数字名 prompt → 生成手写数字」全流程（教学版，非工业画质）

## 本节目录

- 零、9 节学习路线总览
- 一、07.1 Stable Diffusion 在做什么？
- 二、07.2 VAE 与潜空间
- 三、07.3 Text Encoder
- 四、07.4 U-Net 中的 Cross-Attention
- 五、07.5 Classifier-Free Guidance
- 六、07.6 组装完整 LDM
- 七、07.7 训练（VAE → LDM）
- 八、07.8 按 prompt 采样
- 九、07.9 可视化 Cross-Attention
- 十、文件说明与环境
- 十一、与 DDPM / Transformer 的联系

---

## 零、9 节学习路线总览

```
07.1  SD 直觉：像素太贵 → 潜空间 + 文本条件
  ↓
07.2  VAE：x ↔ z
  ↓
07.3  Text Encoder：prompt → context
  ↓
07.4  Cross-Attn：图像 Q 查文本 K/V（对照 06.6）
  ↓
07.5  CFG：无条件外推，加强「听话」
  ↓
07.6  组装 MiniLDM，验维度
  ↓
07.7  训练 VAE → 潜空间条件扩散
  ↓
07.8  推理：prompt → 图像
  ↓
07.9  画出 Cross-Attn 热力图
```

| 小节 | 脚本 | 核心问题 | 预计耗时 |
|------|------|----------|----------|
| 07.1 | `01_sd_intro.py` | SD 相对像素 DDPM 解决了什么？ | 1 分钟 |
| 07.2 | `02_vae_latent.py` | 潜空间怎么压缩 / 重建？ | 1 分钟 |
| 07.3 | `03_text_encoder.py` | 文字怎么变成向量？ | 1 分钟 |
| 07.4 | `04_cross_attn_unet.py` | 文本如何注入 U-Net？ | 1 分钟 |
| 07.5 | `05_cfg.py` | guidance scale 是什么？ | 1 分钟 |
| 07.6 | `06_model_assembly.py` | 整条链路 shape 对不对？ | 1 分钟 |
| 07.7 | `07_train.py` | 怎么训出能听 prompt 的模型？ | CPU `--fast` 约数分钟～十余分钟 |
| 07.8 | `08_sample.py` | 只有文字时怎么出图？ | 视 T 而定 |
| 07.9 | `09_visualize_cross_attn.py` | 模型在「看」哪个词？ | 1～数分钟 |

```bash
conda activate ddpm_learn
cd section07_stable_diffusion

python 01_sd_intro.py
python 02_vae_latent.py
# … 依次到 06 …
python 07_train.py --fast
python 08_sample.py --prompt three
python 09_visualize_cross_attn.py --prompt three
```

> **重要**：本节是 **SD 结构教学版**。真实 Stable Diffusion 用更大 VAE/CLIP/U-Net 与海量图文对；这里用 MNIST + 数字名，只为把管线跑通、把概念对齐。

---

## 一、07.1 Stable Diffusion 在做什么？

### 1.1 像素 DDPM 的瓶颈

Section 05 在 **像素** $x$ 上做 1000 步去噪。分辨率升高后，U-Net 的算力与显存急剧上升。

### 1.2 Latent Diffusion 的想法

```
真实图 x  ──VAE Encoder──►  z0（更小）
z_T ~ N(0,I) ──条件 U-Net 去噪──► z0
z0 ──VAE Decoder──► 生成图 x̂
```

扩散只在潜变量 $z$ 上进行 → 网络输入空间更小 → 才能做高分辨率文生图。

### 1.3 三大件对照

| 组件 | 真实 Stable Diffusion | 本仓库迷你版 |
|------|----------------------|--------------|
| VAE | KL-VAE，约 8× 空间下采样 | TinyVAE，28→7（约 4×） |
| Text Encoder | CLIP | 数字名词嵌入 + 浅层 Transformer |
| Cond U-Net | Cross-Attn U-Net | CondUNet（4×7×7） |
| 引导 | CFG | 同左 |

运行：`python 01_sd_intro.py`

---

## 二、07.2 VAE 与潜空间

### 2.1 压缩比（本项目）

| 张量 | shape | 元素数 |
|------|-------|--------|
| 像素 $x$ | $(B,1,28,28)$ | 784 |
| 潜变量 $z$ | $(B,4,7,7)$ | 196 |

约 **4×** 压缩。真实 SD 常见 $512\to 64$ 的空间下采样（再乘通道）。

### 2.2 训练目标

$$\mathcal{L}_{\mathrm{VAE}} = \|x - \hat{x}\|^2 + \lambda\,\mathrm{KL}\big(q(z|x)\,\|\,\mathcal{N}(0,I)\big)$$

本仓库默认 $\lambda=10^{-4}$（轻量 KL，优先重建）。

### 2.3 与扩散的配合

- **训 VAE**：可学习 `reparameterize`
- **训 LDM / 采样**：常用 **确定性** `encode → μ`，再 `decode(z0)`

运行：`python 02_vae_latent.py`（有 checkpoint 时会显示清晰重建）

---

## 三、07.3 Text Encoder

### 3.1 接口（与 CLIP 同构）

```
"three"  →  token ids  →  TextEncoder  →  context (B, L, D)
```

本教学版词表：

```
<pad>, <uncond>, zero, one, …, nine
```

`<uncond>` 专供 CFG：训练时随机替换；采样时作为无条件分支。

### 3.2 对照

| | CLIP（真实 SD） | 本教学版 |
|--|-----------------|----------|
| 分词 | BPE，最长约 77 | 空格切，长度 1 |
| 输出 | $(B,77,768)$ 量级 | $(B,1,64)$ |

运行：`python 03_text_encoder.py`

---

## 四、07.4 U-Net 中的 Cross-Attention

### 4.1 与 Section 06.6 的同构

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

| | 机器翻译（06.6） | 文生图（本节） |
|--|-----------------|---------------|
| Q | Decoder 译文 | 图像（潜空间）特征 |
| K, V | Encoder memory | 文本 context |
| 对齐矩阵 | $(L_{\mathrm{tgt}}, L_{\mathrm{src}})$ | $(H\cdot W,\ L_{\mathrm{text}})$ |

### 4.2 在 CondUNet 中的位置

```
z_t + t_emb
  → Down（7×7 → 4×4）
  → Mid Res
  → Cross-Attn(Q=图像, K/V=文本)   ← 本节重点
  → Mid Res
  → Up + Skip → ε_θ
```

运行：`python 04_cross_attn_unet.py`

---

## 五、07.5 Classifier-Free Guidance

### 5.1 公式

$$\hat\varepsilon = \varepsilon_u + s\cdot(\varepsilon_c - \varepsilon_u)$$

| 符号 | 含义 |
|------|------|
| $\varepsilon_c$ | 有 prompt 时的噪声预测 |
| $\varepsilon_u$ | `<uncond>` 时的噪声预测 |
| $s$ | `guidance_scale`，越大越跟 prompt，过大易过饱和 |

### 5.2 训练侧

`collate_with_cfg_drop`：约 10% 样本把文本换成 `<uncond>`，使同一网络学会两条分支。

运行：`python 05_cfg.py`

---

## 六、07.6 组装完整 LDM

数据流：

```
prompt → TextEncoder → context
x → VAE Enc → z0 → q_sample → z_t → CondUNet → ε_θ
采样得到 z0 → VAE Dec → x̂
```

运行：`python 06_model_assembly.py`（确认 $\varepsilon$ 与 $z_t$ 同形）

---

## 七、07.7 训练

### 7.1 两阶段

1. **VAE**：重建 MNIST  
2. **LDM**：冻结 VAE；在 $z$ 上优化 $L_{\mathrm{simple}}=\|\varepsilon-\varepsilon_\theta(z_t,t,c)\|^2$，并做 CFG drop

### 7.2 命令

```bash
# 推荐第一次：一键 fast
python 07_train.py --fast

# 分阶段
python 07_train.py --stage vae --epochs 8
python 07_train.py --stage ldm --epochs 15

# 自定义扩散步数（采样更慢但更细）
python 07_train.py --stage ldm --timesteps 200
```

`--fast`：子集 5000、VAE≈3 epoch、LDM≈5 epoch、`T=30`，便于 CPU 冒烟（约 1～2 分钟）。

### 7.3 输出

| 路径 | 内容 |
|------|------|
| `checkpoints/vae_last.pth` | VAE 权重 |
| `checkpoints/ldm_last.pth` | 完整 MiniLDM |
| `figures/vae_recon_*.png` | VAE 重建 |
| `figures/ldm_samples_epoch_*.png` | 训练中采样 |
| `figures/ldm_loss_curve.png` | LDM 损失曲线 |

---

## 八、07.8 按 prompt 采样

```bash
python 08_sample.py --prompt three
python 08_sample.py --prompt seven --guidance-scale 5
python 08_sample.py --all-digits
```

推理步骤：

1. `TextEncoder(prompt)`  
2. $z_T\sim\mathcal{N}(0,I)$  
3. 逐步 CFG 去噪  
4. `VAE.decode(z0)`

合法 prompt：`zero` … `nine`（见 `data_utils.DIGIT_NAMES`）。

---

## 九、07.9 可视化 Cross-Attention

```bash
python 09_visualize_cross_attn.py --prompt three
```

左图：生成结果；右图：某时间步 bottleneck 上空间位置对文本 token 的注意力（头平均）。

---

## 十、文件说明与环境

| 文件 / 目录 | 说明 |
|-------------|------|
| `01_sd_intro.py` … `09_visualize_cross_attn.py` | 分节脚本 |
| `sd_model.py` | TinyVAE / TextEncoder / CondUNet / Schedule |
| `data_utils.py` | MNIST + 数字名词表 + CFG drop |
| `checkpoints/` | 权重（gitignore） |
| `figures/` | 可视化（gitignore） |
| `data/` | MNIST（gitignore） |

```bash
conda activate ddpm_learn
cd section07_stable_diffusion
```

依赖与 Section 05 相同：`torch`, `torchvision`, `matplotlib`, `tqdm`（无需额外下载 SD 权重）。

### 一键顺序跑通

```bash
python 01_sd_intro.py
python 02_vae_latent.py
python 03_text_encoder.py
python 04_cross_attn_unet.py
python 05_cfg.py
python 06_model_assembly.py
python 07_train.py --fast
python 08_sample.py --prompt three
python 09_visualize_cross_attn.py --prompt three
```

---

## 十一、与 DDPM / Transformer 的联系

| 概念 | Section 01~05 DDPM | Section 06 Transformer | Section 07 迷你 SD |
|------|--------------------|------------------------|-------------------|
| 生成空间 | 像素 $x$ | 离散 token | 潜变量 $z$ 再解码 |
| 条件 | 时间 $t$ | 源句 | 时间 $t$ + 文本 |
| Cross-Attn | （无文本） | 译文 Q ← 原文 K/V | 图像 Q ← 文本 K/V |
| 损失 | $\|\varepsilon-\varepsilon_\theta\|^2$ | CrossEntropy | 同左（在 $z$ 上） |
| 采样增强 | — | Beam 等 | **CFG** |

Section 04 README 曾提醒：U-Net 里的 Self-Attn **不是** SD 的文本 Cross-Attn——本节补上后者。

---

## 课程衔接

| 章节 | 内容 |
|------|------|
| Section 01~05 | 像素 DDPM 全流程 |
| Section 06 | Transformer + Cross-Attention |
| **Section 07** | **潜空间 + 文本条件 + CFG = 迷你 Stable Diffusion** |

恭喜你把「扩散生成」与「注意力条件控制」接到了同一条文生图管线上。
