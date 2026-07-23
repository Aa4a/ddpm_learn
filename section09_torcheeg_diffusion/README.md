# Section 09 - TorchEEG 中的扩散模型（架构与原理）

> 文档专节：只讲原理与 API 架构，**不要求跑通训练**。  
> 库文档：[torcheeg.readthedocs.io](https://torcheeg.readthedocs.io/) · 扩散相关：`BUNet` / `BCUNet` + `DDPMTrainer` / `CDDPMTrainer`

## 本节目标

- 弄清 TorchEEG 把 **EEG 生成** 接到 **DDPM** 上时，数据长什么样
- 分清 **模型（BUNet / BCUNet）** 与 **训练器（DDPMTrainer / CDDPMTrainer）** 各自管什么
- 对照本仓库 §01–05：同一套加噪 / 预测 ε / 反向采样，换了「EEG 网格」输入与封装方式
- 知道条件生成时 **类别 y** 如何注入（相对 §07 CFG 的简化版）

---

## 零、先看一张总图

```
真实 EEG（预处理后）
        │
        ▼
   网格张量 x0 : (B, C, H, W)     典型 (B, 4, 9, 9)
        │
        │  DDPMTrainer / CDDPMTrainer 内部
        │  ┌─────────────────────────────────────┐
        │  │  t ~ Uniform{1..T}                  │
        │  │  ε ~ N(0,I)                         │
        │  │  xt = √ᾱ_t x0 + √(1-ᾱ_t) ε          │  ← 你已在 §01 学过
        │  │  ε̂ = UNet(xt, t [, y])              │  ← BUNet / BCUNet
        │  │  L ≈ ||ε - ε̂||²                     │  ← §03 L_simple
        │  └─────────────────────────────────────┘
        │
采样时：x_T ~ N(0,I) ──逐步 p_sample──► x0̂（合成 EEG 网格）
```

| 角色 | TorchEEG 组件 | 对应本课程 |
|------|---------------|------------|
| 噪声预测网络 | `BUNet`（无条件）/ `BCUNet`（类别条件） | §04 U-Net + 时间嵌入 |
| 扩散日程 + 损失 + 采样 | `DDPMTrainer` / `CDDPMTrainer` | §01–03 公式 + §05 训练循环 |
| 数据形态 | 通道特征 × 电极空间网格 | 不是原始 1D 时序波形 |

官方也写明：这两套 U-Net **并非专为 EEG 设计**，而是把图像 DDPM 基线接到「网格化 EEG」上，作为生成研究起点。

---

## 一、EEG 在 TorchEEG 里为什么是 `(C, 9, 9)`？

### 1.1 从多通道时序到「类图像」

原始 EEG 常是：

```
(电极数, 时间点)  或  (频带特征, 电极)
```

TorchEEG 情感识别管线里常见做法（如 DEAP）：

1. 对每导联提频带特征（如差分熵 Band Differential Entropy）→ 得到 **C 个特征通道**（默认示例常取 **4 个频带**）
2. `ToGrid`：按电极的 **二维头皮位置** 把特征摆进 `H×W` 网格（默认 **9×9**）
3. 得到张量形状：

```
x ∈ R^{C × H × W}   例如  4 × 9 × 9
```

于是可以直接套用 **2D U-Net + 图像扩散** 那一套：把「空间相邻电极」当成「像素邻域」。

### 1.2 和本仓库 MNIST DDPM 的对照

| | §05 MNIST DDPM | TorchEEG 扩散 |
|--|----------------|---------------|
| `x0` 含义 | 手写数字图像 | 电极网格上的频带特征图 |
| 典型 shape | `(1, 28, 28)` | `(4, 9, 9)` |
| 空间维 | 像素 | 头皮电极布局 |
| 通道维 | 灰度 | 频带 / 特征维 |
| 数学 | 同一套高斯 DDPM | 同一套 |

**要点**：TorchEEG 扩散生成的是「预处理后的网格特征」，不是示波器上的原始电压波形。下游若要「像真 EEG」，还依赖预处理是否可逆、任务是否只需要特征分布匹配。

---

## 二、模型层：BUNet 与 BCUNet

### 2.1 BUNet（无条件）

```text
输入:  xt (B, C, H, W),  t (B,)
输出:  ε̂ 或 与 xt 同形的预测  (B, C, H, W)
```

API 直觉：

```python
from torcheeg.models import BUNet

unet = BUNet(in_channels=4, hid_channels=64, grid_size=(9, 9), beta_timesteps=256)
eps_pred = unet(xt, t)   # xt: (B,4,9,9), t: (B,)
```

| 参数 | 含义 |
|------|------|
| `in_channels` | 特征通道 C（频带数等） |
| `hid_channels` | U-Net 基础宽度 |
| `grid_size` | `(H, W)`，需与数据网格一致 |
| `beta_timesteps` | 模型侧与扩散步相关的配置（与 Trainer 的 `timesteps` 配合使用） |

结构角色（对照 §04）：

```
xt ──► 时间嵌入(t) 注入各尺度残差块
   ──► Down / Mid / Up（含 Attention 的 U-Net）
   ──► 预测噪声（或 x0 / v，由 Trainer 的 parameterization 决定）
```

### 2.2 BCUNet（类别条件）

在 BUNet 上增加 **离散标签 y**（如 valence 高低、情绪类别）：

```text
输入:  xt (B,C,H,W),  t (B,),  y (B,)
输出:  同形预测
```

```python
from torcheeg.models import BCUNet

unet = BCUNet(in_channels=4, num_classes=2, grid_size=(9, 9))
eps_pred = unet(xt, t, y)
```

条件如何起作用（原理层面）：

- 把类别 id 做成 **embedding**
- 与时间嵌入一起（或经 FiLM / 相加 / 拼接）注入 U-Net 各层
- 去噪轨迹被「拉向」该类 EEG 的条件分布 `p(x₀ | y)`

与 §07 CFG 的差别：

| | §07 迷你 SD | TorchEEG BCUNet |
|--|-------------|-----------------|
| 条件 | 文本 embedding + Cross-Attn | 类别 embedding |
| 引导 | 常配 CFG（有/无条件外推） | Trainer 侧直接条件去噪（是否做 CFG 取决于用法，默认是标签条件 DDPM） |

---

## 三、训练器层：DDPM 公式装在哪里？

模型只负责 `f(xt, t[, y])`。  
**β schedule、q_sample、损失、反向采样** 在 `DDPMTrainer` / `CDDPMTrainer`（PyTorch Lightning 模块）里。

### 3.1 与你已学公式的一一对应

正向（Trainer 内 `q_sample`）：

```
x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
```

默认训练目标（`parameterization="eps"`）：

```
L_simple = E_{t,x₀,ε} [ ||ε - ε_θ(x_t, t[, y])||² ]
```

也可选 `"x0"` / `"v"` 参数化；还可加 ELBO 加权项（`original_elbo_weight`、`lvlb_weights`），默认以 **simple MSE** 为主（`l_simple_weight=1`）。

反向单步（`p_sample`）：由网络预测还原 `x₀`（或等价地由 ε 反推），再取后验均值，并在 `t>0` 时加噪声——与 §02 / §05 相同。

### 3.2 两个 Trainer 的分工

| | `DDPMTrainer` | `CDDPMTrainer` |
|--|---------------|----------------|
| 配套模型 | `BUNet` | `BCUNet` |
| batch | `(x,)` 或 `(x, y)` 中 y 不用 | `(x, y)`，y 传入网络 |
| 学的分布 | `p(x)` | `p(x|y)` |
| 采样 | 无标签 | 指定类别生成 |

用法骨架（理解用，非本节必跑）：

```python
from torcheeg.models import BUNet, BCUNet
from torcheeg.trainers import DDPMTrainer, CDDPMTrainer

# 无条件
model = BUNet(in_channels=4)
trainer = DDPMTrainer(model, timesteps=1000, beta_schedule="linear",
                      accelerator="cpu")  # 或 gpu
# trainer.fit(train_loader, val_loader)

# 条件
cmodel = BCUNet(in_channels=4, num_classes=2)
ctrainer = CDDPMTrainer(cmodel, timesteps=1000, accelerator="cpu")
# ctrainer.fit(train_loader, val_loader)
```

### 3.3 Trainer 里你值得知道的超参

| 超参 | 默认直觉 | 与本课程关系 |
|------|----------|--------------|
| `timesteps` | 1000 | 扩散步数 T |
| `beta_schedule` | `"linear"`；还有 cosine / sqrt_linear / sqrt | §01 β schedule |
| `loss_type` | `"l2"`（也可 l1） | §03 MSE |
| `parameterization` | `"eps"` | 预测噪声 |
| `use_ema` | True | 指数滑动平均权重，采样更稳 |
| `linear_start` / `linear_end` | 1e-4 / 2e-2 | 与经典 DDPM 一致 |
| `metrics` | 可选 `fid` / `is` | 生成质量；需额外特征提取器，且常只在 test 算 |

数据加载约定：与 TorchEEG 其它任务一样，`DataLoader` 产出 `(eeg_tensor, label)`；无条件训练时标签可忽略，条件训练时必须与 `num_classes` 一致。

---

## 四、整条「研究向」流水线（概念）

真实项目里通常是：

```
原始 DEAP / SEED / ... 
  → transforms（频带特征、BaselineRemoval、ToGrid、ToTensor）
  → Dataset / KFold
  → DataLoader → (x: 4×9×9, y: 类别)
  → CDDPMTrainer + BCUNet.fit
  → sample → 合成网格 EEG
  →（可选）用分类器 / FID 评估「像不像该类」
```

教学上只需记住三层：

1. **表示**：EEG → 网格张量  
2. **去噪器**：BUNet / BCUNet  
3. **扩散引擎**：DDPMTrainer / CDDPMTrainer  

---

## 五、架构对照总表

```
┌──────────────────────────────────────────────────────────┐
│                    TorchEEG 扩散栈                        │
├─────────────────┬────────────────┬───────────────────────┤
│ 数据表示         │ 去噪网络        │ 扩散过程              │
│ ToGrid 等        │ BUNet/BCUNet   │ DDPMTrainer/...       │
│ (C,H,W) 张量     │ ε_θ(xt,t[,y])  │ q_sample / p_sample   │
├─────────────────┴────────────────┴───────────────────────┤
│              数学内核 = 经典连续 DDPM（§01–05）            │
└──────────────────────────────────────────────────────────┘
```

| 你已掌握 | TorchEEG 落点 |
|----------|----------------|
| `x_t = √ᾱ x₀ + √(1-ᾱ) ε` | Trainer.`q_sample` |
| `L = ||ε - ε_θ||²` | Trainer.`p_losses`（eps 模式） |
| 时间嵌入 + U-Net | BUNet / BCUNet 内部 |
| 类别 / 文本条件 | BCUNet 的 y（文本 Cross-Attn 不在此默认栈里） |
| CFG | §07 概念；TorchEEG 默认文档路径是直接条件 DDPM |

---

## 六、读源码 / 文档时的导航

| 想搞懂… | 去看 |
|---------|------|
| 网络输入输出 shape | [BUNet](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.models.BUNet.html) / [BCUNet](https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.models.BCUNet.html) |
| schedule、损失、采样 | `torcheeg.trainers` 中 `DDPMTrainer` / `CDDPMTrainer`（Lightning） |
| Trainer 总览 | [examples_trainers](https://torcheeg.readthedocs.io/en/latest/auto_examples/examples_trainers.html)（生成模型小节） |
| 网格从哪来 | `transforms.ToGrid` + 各数据集的 `CHANNEL_LOCATION_DICT` |

---

## 七、常见误解

1. **「生成了原始脑电波」** — 默认生成的是 **网格特征图**；是否还原到时序取决于特征是否可逆。  
2. **「BUNet = 专为 EEG 发明的扩散」** — 官方定位是 **图像 DDPM 基线迁移到网格 EEG**。  
3. **「有 BCUNet 就等于有 SD 那种文本控」** — 默认是 **类别条件**，不是 CLIP 文本。  
4. **「装上 torcheeg 就自动学会扩散」** — 网络与 Trainer 分离；数学仍是你在 §01–05 学的那一套。

---

## 八、和前后章节怎么接

| 章节 | 关系 |
|------|------|
| §01–05 | 数学与训练循环的「内核」；本节只换数据与库封装 |
| §07 | 条件生成更复杂（文本 + CFG）；TorchEEG 默认更简单（类别） |
| §08 | 离散 token 扩散；EEG 这条线仍是 **连续高斯** 扩散 |

**一句话**：TorchEEG 扩散 = **把 EEG 摆成小图像** + **标准 DDPM U-Net** + **Lightning Trainer 托管 schedule/损失/采样**；条件版多一个类别嵌入。

---

## 九、若以后要动手（可选备忘）

不强制；需要实验时再：

1. 安装 `torcheeg`（依赖含 `pytorch-lightning`、`mne` 等；Windows 注意用预编译 `scipy`）  
2. 用合成 `(B,4,9,9)` 或小型公开集验证 `BUNet(xt,t)` shape  
3. 再接到 `DDPMTrainer.fit`；条件任务换 `BCUNet` + `CDDPMTrainer`

本节正文到此为止，**以架构与原理对齐为目标**。
