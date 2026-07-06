# Section 04 - U-Net 架构：时间嵌入与残差去噪网络

## 本节目标

- 理解为什么去噪网络 $\varepsilon_\theta(x_t, t)$ 需要输入时间步 $t$
- 掌握正弦时间嵌入（Sinusoidal Position Embedding）的数学原理与实现
- 掌握如何在残差块（Residual Block）中注入时间信息
- 理解自注意力（Self-Attention）的数学公式，以及与 Cross-Attention 的区别
- 掌握 U-Net 三大核心操作：下采样、跳跃连接、上采样的原理与尺寸变化
- 学习并组装一个完整的简易去噪 U-Net 网络，验证其输入输出维度一致性

## 本节目录

- 零、为什么去噪网络需要时间步 $t$
- 一、正弦时间嵌入（Sinusoidal Position Embedding）
- 二、时间注入残差块（Residual Block with Time Embedding）
- 三、自注意力机制（Self-Attention）
- 四、U-Net 核心操作：下采样、跳跃连接、上采样
- 五、简易 U-Net 架构拼装与维度追踪
- 六、维度验证与测试

---

## 零、为什么去噪网络需要时间步 $t$

在前面的章节中，我们知道 DDPM 的训练目标是最小化简化损失：

$$L_{\rm simple} = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(x_t, t)\|^2\right]$$

去噪网络 $\varepsilon_\theta$ 的输入不仅有带噪图像 $x_t$，还有**时间步 $t$**。为什么必须输入 $t$？

1. **共享参数的需要**：整个去噪过程（$T=1000$ 步）共用**同一个**神经网络 $\varepsilon_\theta$。如果不输入 $t$，网络就无法得知当前输入的 $x_t$ 处于哪一个加噪阶段（是刚开始去噪，还是快接近纯噪声了）。
2. **去噪任务的差异性**：
   - 在 $t$ 很大时（如 $t \approx 1000$），$x_t$ 几乎是纯高斯噪声，网络需要预测大尺度的、方向性的粗糙噪声，以重建图像的大体轮廓。
   - 在 $t$ 很小时（如 $t \approx 1$），$x_t$ 已经非常接近真实图像，网络只需要预测微小的、局部的细节噪声，进行高频边缘的微调。
   - **不同的 $t$ 对应完全不同的去噪任务**。因此，必须将 $t$ 传入网络，作为强有力的条件（Conditioning）来引导网络进行针对性的去噪。

---

## 一、正弦时间嵌入（Sinusoidal Position Embedding）

### 1.1 数学原理

为了将标量时间步 $t \in [0, T-1]$ 输入到神经网络中，我们需要将其转换为高维连续向量。DDPM 采用了 Transformer 中经典的正弦位置编码（Sinusoidal Position Encoding）：

$$PE(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right)$$
$$PE(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

其中：
- $t$ 是当前的时间步
- $d$ 是输出的时间嵌入维度（例如 $d=128$）
- $i \in [0, d/2 - 1]$ 是维度的索引

### 1.2 为什么使用正弦嵌入？

1. **外推性与连续性**：正弦和余弦函数具有天然的周期性和连续性。即使在训练中没有见过某些时间步，网络也能通过插值和外推合理地理解它们。
2. **相对位置关系**：对于任意固定的偏移量 $k$，$PE(t+k)$ 可以表示为 $PE(t)$ 的线性变换。这使得网络非常容易学习到时间步之间的相对先后顺序和距离。

### 1.3 代码实现

在 `01_unet_components.py` 中，我们通过指数衰减因子来实现这一公式：

```python
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # 计算 10000^(-2i/dim) 的对数指数衰减形式
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 外积计算
        embeddings = time[:, None] * embeddings[None, :]
        # 拼接 sin 和 cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

---

## 二、时间注入残差块（Residual Block with Time Embedding）

### 2.1 架构设计

仅仅得到时间嵌入 $t_{\rm emb} \in \mathbb{R}^d$ 还不够，我们需要将它“注入”到图像特征图 $h \in \mathbb{R}^{B \times C \times H \times W}$ 中。

在残差块中，注入的步骤如下：
1. **时间投影**：通过一个多层感知机（MLP，通常是 SiLU + Linear）将时间嵌入从维度 $d$ 投影到与当前特征图相同的通道数 $C$。
2. **维度对齐**：将投影后的时间向量形状从 $[B, C]$ 扩展为 $[B, C, 1, 1]$。
3. **特征相加**：将时间信息直接加到第一层卷积输出的特征图上：$h = h + t_{\rm proj}$。

```
特征图输入 x ──> [ Conv1 ] ──> ( + ) ──> [ Conv2 ] ──> ( + ) ──> 输出
                                 ^                      ^
                                 │                      │
时间嵌入 t ───> [ MLP ] ─────────┘                      │
                                                        │
残差连接 ───────────────────────────────────────────────┘
```

### 2.2 代码实现

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]  # 扩展维度对齐
        h = h + time_emb                              # 时间注入
        h = self.conv2(h)
        return h + self.shortcut(x)
```

---

## 三、自注意力机制（Self-Attention）

> **注意**：这里的注意力是 **Self-Attention（自注意力）**——特征图上的每个位置与同一张图内的其他位置建立联系。它**不是** Stable Diffusion 里那种 **Cross-Attention（文本→图像）** 的条件注意力。原版 DDPM 是无条件生成模型，只有时间步 $t$ 作为条件，没有文本输入。

### 3.1 为什么需要注意力机制？

卷积神经网络（CNN）的感受野是局部的。$3 \times 3$ 卷积核在单层内只能看到 9 个邻域像素。虽然堆叠多层可以间接扩大感受野，但长距离依赖（如手写数字的横笔与竖笔）仍然难以高效建模。

在 U-Net 的**低分辨率层**（如 $8 \times 8$）引入自注意力，可以让每个位置**直接**与所有 $N = H \times W$ 个位置交互，计算复杂度为 $O(N^2)$。由于 $N$ 很小（64），开销可控。

| 机制 | 每个位置能"看到"的范围 | 计算代价 |
|------|----------------------|----------|
| $3 \times 3$ 卷积 | 局部 9 像素 | $O(HW)$ |
| 堆叠 $L$ 层卷积 | 约 $(2L+1)^2$ 像素（间接） | $O(L \cdot HW)$ |
| Self-Attention | **全部** $HW$ 个位置（直接） | $O((HW)^2)$，仅在低分辨率使用 |

### 3.2 数学原理：Scaled Dot-Product Attention

将特征图 $x \in \mathbb{R}^{B \times C \times H \times W}$ 展平为 $N = H \cdot W$ 个空间 token，每个 token 是 $C$ 维向量。

**Step 1：线性投影得到 Q、K、V**

$$Q = x W_Q, \quad K = x W_K, \quad V = x W_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{C \times C}$。代码里用 $1 \times 1$ 卷积一次算出 $3C$ 个通道再拆分：

$$[Q;\, K;\, V] = \text{Conv}_{1 \times 1}(x), \quad Q, K, V \in \mathbb{R}^{B \times C \times H \times W}$$

**Step 2：计算注意力权重**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{C}}\right) V$$

- $Q K^\top$ 的形状为 $[B,\, N,\, N]$：第 $(i,j)$ 元素表示位置 $i$ 对位置 $j$ 的相关程度
- 除以 $\sqrt{C}$（缩放因子）防止内积过大导致 softmax 梯度消失
- softmax 按行归一化，使每个位置对所有位置的权重之和为 1

**Step 3：残差连接**

$$\text{output} = x + \text{Proj}(\text{Attention}(Q, K, V))$$

残差保证：即使注意力层初始时学不好，原始特征 $x$ 仍可通过捷径传递。

### 3.3 与 Cross-Attention 的区别（常见误解）

| | Self-Attention（本节） | Cross-Attention（文生图） |
|--|------------------------|---------------------------|
| Q 来自 | 图像特征图 | 图像特征图 |
| K, V 来自 | **同一张**图像特征图 | **文本** embedding |
| 注意力矩阵 | $[N,\, N]$（图像内部） | $[N,\, L]$（$L$ = 文本 token 数） |
| 作用 | 理解图像全局结构 | 让生成内容符合文字描述 |

### 3.4 代码实现

```python
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gn = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.gn(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # 展平空间维度
        q = q.view(B, C, H * W).transpose(1, 2)  # [B, N, C]
        k = k.view(B, C, H * W)                  # [B, C, N]
        v = v.view(B, C, H * W).transpose(1, 2)  # [B, N, C]

        # Scaled Dot-Product Attention
        attn = torch.bmm(q, k) * (C ** -0.5)     # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return x + self.proj(out)
```

---

## 四、U-Net 核心操作：下采样、跳跃连接、上采样

在拼装完整 U-Net 之前，必须先理解 U 形网络的三个核心操作。它们共同解决一个矛盾：

> **要看全局结构，就必须缩小分辨率；要输出逐像素噪声，就必须恢复原始分辨率。**

```
编码器：分辨率↓ 语义↑  （看大局）
解码器：分辨率↑ 细节↑  （画回来）
跳跃连接：把编码器的高分辨率细节直接传给解码器（防止变糊）
```

---

### 4.1 下采样（Downsampling）

#### 4.1.1 做什么？

将特征图的空间尺寸减半：$H \times W \;\to\; \frac{H}{2} \times \frac{W}{2}$，同时通常增加通道数（提取更丰富的语义特征）。

#### 4.1.2 数学原理

本代码使用 **stride = 2 的卷积**（而非最大池化）：

$$\text{Conv2d}(x)_{i,j,c'} = \sum_{c}\sum_{u=0}^{3}\sum_{v=0}^{3} W_{c',c,u,v} \cdot x_{2i+u,\; 2j+v,\; c} + b_{c'}$$

- `kernel_size=4, stride=2, padding=1`：输出尺寸公式为

$$\boxed{H_{\rm out} = \left\lfloor \frac{H_{\rm in} + 2p - k}{s} \right\rfloor + 1 = \left\lfloor \frac{32 + 2 - 4}{2} \right\rfloor + 1 = 16}$$

- 每 2 个输入像素合并为 1 个输出像素 → 分辨率减半
- 通道数可以从 16 变为 32（`down2_res` 中完成），下采样层本身保持通道不变

#### 4.1.3 为什么要下采样？

1. **扩大感受野**：在 $8 \times 8$ 的特征图上，一个 $3 \times 3$ 卷积"看到"的原图区域比在 $32 \times 32$ 上大得多
2. **提取高层语义**：浅层捕捉边缘/纹理，深层捕捉形状/结构
3. **降低计算量**：Self-Attention 的复杂度是 $O(N^2)$，$N=64$ 远比 $N=1024$ 可行

#### 4.1.4 代码对应

```python
self.down1_pool = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1)  # 32→16
self.down2_pool = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)  # 16→8
```

---

### 4.2 跳跃连接（Skip Connection）

#### 4.2.1 做什么？

将编码器某一层的输出**原样保存**，在解码器对应层通过**通道拼接（concatenate）** 融合：

$$h_{\rm dec} = \text{Concat}\big(h_{\rm up},\; h_{\rm enc}\big), \quad \text{dim=1（通道维）}$$

#### 4.2.2 数学原理

设编码器保存的特征为 $s \in \mathbb{R}^{B \times C_s \times H \times W}$，解码器上采样后的特征为 $u \in \mathbb{R}^{B \times C_u \times H \times W}$，则：

$$h = [u \;\|\; s] \in \mathbb{R}^{B \times (C_u + C_s) \times H \times W}$$

本代码中的两次跳跃连接：

| 跳跃连接 | 编码器保存 | 解码器上采样后 | 拼接结果 |
|----------|-----------|---------------|----------|
| Skip 2 | `x2_res` $[B,32,16,16]$ | `up1_unpool` $[B,32,16,16]$ | $[B,64,16,16]$ |
| Skip 1 | `x1_res` $[B,16,32,32]$ | `up2_unpool` $[B,16,32,32]$ | $[B,32,32,32]$ |

拼接后的 $h$ 送入 `ResidualBlock(in=64, out=16)` 或 `ResidualBlock(in=32, out=16)` 融合两套信息。

#### 4.2.3 为什么是 Concat 而不是 Add？

| 操作 | 公式 | 特点 |
|------|------|------|
| **Add（残差）** | $h = u + s$ | 要求 $C_u = C_s$，两种信息混合相加 |
| **Concat（U-Net）** | $h = [u \,\|\, s]$ | 通道数翻倍，**两套信息完整保留**，由后续卷积学习如何融合 |

U-Net 选择 Concat 是因为编码器（高分辨率细节）和解码器（低分辨率语义）携带**不同类型**的信息，直接相加会互相干扰。

#### 4.2.4 为什么需要跳跃连接？

下采样是有损的：$32 \times 32 \to 8 \times 8$ 丢失了大量空间细节。若解码器只从 $8 \times 8$ 上采样，输出会**模糊、边界不清**。

跳跃连接建立了**信息高速公路**：

```
编码器 x1_res（32×32 的高频细节：边缘、笔画）
         │
         └──────→ 直接传给解码器，与语义特征融合
```

这正是 U-Net（Ronneberger et al., 2015）的核心创新，DDPM 沿用了这一设计。

---

### 4.3 上采样（Upsampling）

#### 4.3.1 做什么？

将特征图的空间尺寸加倍：$\frac{H}{2} \times \frac{W}{2} \;\to\; H \times W$，逐步恢复到与输入相同的分辨率。

#### 4.3.2 数学原理

本代码使用 **转置卷积（Transposed Convolution / ConvTranspose2d）**：

$$\text{ConvTranspose2d}(x)_{i,j,c'} = \sum_{c}\sum_{u,v} W_{c',c,u,v} \cdot x_{\lfloor(i-u)/s\rfloor,\; \lfloor(j-v)/s\rfloor,\; c}$$

- `kernel_size=4, stride=2, padding=1`：输出尺寸公式为

$$\boxed{H_{\rm out} = (H_{\rm in} - 1) \times s - 2p + k = (8 - 1) \times 2 - 2 + 4 = 16}$$

- 可理解为卷积的"逆操作"：在输入像素之间**插入零**再卷积，实现空间放大
- 另一种常见做法是 `nn.Upsample(scale_factor=2)` + 普通卷积，效果类似

#### 4.3.3 为什么需要上采样？

去噪网络 $\varepsilon_\theta(x_t, t)$ 的输出必须与输入**逐像素对齐**：

$$\varepsilon_\theta(x_t, t) \in \mathbb{R}^{B \times C \times H \times W}, \quad \text{与 } x_t \text{ 同形状}$$

Bottleneck 处特征图只有 $8 \times 8$，必须通过上采样逐步恢复到 $32 \times 32$（或 $28 \times 28$ 等原始尺寸）。

#### 4.3.4 代码对应

```python
self.up1_unpool = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  # 8→16
self.up2_unpool = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)  # 16→32
```

---

### 4.4 三者协作：编码器-解码器的信息流

以输入 $x_t \in \mathbb{R}^{B \times 3 \times 32 \times 32}$ 为例，完整的信息流如下：

```
阶段          操作              空间尺寸    通道数    信息类型
─────────────────────────────────────────────────────────────
编码器        init_conv         32×32       16       原始特征
              down1_res         32×32       16       浅层细节 → 存为 x1_res ★
              down1_pool        16×16       16       第一次下采样
              down2_res         16×16       32       中层特征 → 存为 x2_res ★
              down2_pool         8×8        32       第二次下采样
Bottleneck    mid_res + attn     8×8        32       全局语义理解
解码器        up1_unpool        16×16       32       第一次上采样
              cat(x2_res)       16×16       64       融合中层细节 ★
              up1_res           16×16       16       特征精炼
              up2_unpool        32×32       16       第二次上采样
              cat(x1_res)       32×32       32       融合浅层细节 ★
              up2_res           32×32       16       特征精炼
输出          out_conv          32×32        3       预测噪声 ε_θ
```

★ 标记的是跳跃连接传递的特征。

---

## 五、简易 U-Net 架构拼装与维度追踪

### 5.1 整体架构图

`SimpleUNet` 采用经典的**沙漏型（Hourglass）**结构：左侧编码器下采样，右侧解码器上采样，横向跳跃连接传递细节。

```
输入 x_t [B,3,32,32] + 时间步 t [B]
         │
         ├──→ time_embed(t) ──→ t_emb [B,128] ──→ 传入每个 ResidualBlock
         │
         ▼
    [ Init Conv ] ──────────────────────────────────────────────→ (cat) ──→ [ Up2 Res ] ──→ [ Out ] ──→ ε_θ
         │                                                            ▲
     [ Down1 Res ] = x1_res ──────────────────────→ (cat) ──→ [ Up1 Res ] │
         │                                            ▲                   │
     [ Down1 Pool ]                              [ Up1 Unpool ]           │
         │                                            │                   │
     [ Down2 Res ] = x2_res ──→ [ Down2 Pool ] ──→ [ Bottleneck ] ───────┘
                                                    (Res+Attn+Res)
```

### 5.2 核心前向传播逻辑

```python
def forward(self, x, t):
    # 1. 计算时间嵌入（只算一次，所有残差块复用）
    t_emb = self.time_embed(t)
    
    # 2. 编码器 (Downsampling)
    x1 = self.init_conv(x)
    x1_res = self.down1_res(x1, t_emb)          # 保存用于跳跃连接 1
    x1_pool = self.down1_pool(x1_res)
    
    x2_res = self.down2_res(x1_pool, t_emb)     # 保存用于跳跃连接 2
    x2_pool = self.down2_pool(x2_res)
    
    # 3. 中间层 (Bottleneck)
    h = self.mid_res1(x2_pool, t_emb)
    h = self.mid_attn(h)
    h = self.mid_res2(h, t_emb)
    
    # 4. 解码器 (Upsampling + Skip Connections)
    h = self.up1_unpool(h)
    h = torch.cat((h, x2_res), dim=1)           # 拼接跳跃连接 2
    h = self.up1_res(h, t_emb)
    
    h = self.up2_unpool(h)
    h = torch.cat((h, x1_res), dim=1)           # 拼接跳跃连接 1
    h = self.up2_res(h, t_emb)
    
    # 5. 输出投影
    return self.out_conv(h)
```

### 5.3 设计要点总结

| 组件 | 在 SimpleUNet 中出现的位置 | 作用 |
|------|--------------------------|------|
| 时间嵌入 | 网络入口，传入每个 ResidualBlock | 告诉网络当前去噪阶段 $t$ |
| 下采样 ×2 | 编码器 | 看全局、提语义、省算力 |
| Self-Attention ×1 | Bottleneck（$8 \times 8$） | 建立长距离像素依赖 |
| 跳跃连接 ×2 | 编码器→解码器 | 保留高分辨率空间细节 |
| 上采样 ×2 | 解码器 | 恢复与输入相同的分辨率 |
| 输出卷积 | 最后一层 $1 \times 1$ Conv | 映射回图像通道数（预测 $\varepsilon$） |

---

## 六、维度验证与测试

在扩散模型中，**去噪网络的输出形状必须与输入图像形状完全一致**（因为我们要对图像中的每一个像素预测其混入的噪声）。

运行本节的测试脚本，可以验证时间嵌入的生成、时间注入残差块、自注意力块以及整个 U-Net 的前向传播是否正常：

```bash
python 01_unet_components.py
```

### 预期输出

```
============================================================
DDPM U-Net 组件与架构测试
============================================================
1. 测试时间嵌入 (Sinusoidal Position Embedding):
   输入时间步 t: [0.0, 100.0, 500.0, 999.0]
   输出嵌入形状: torch.Size([4, 128]) (预期: [4, 128])
   数值范围: 最小值=-1.0000, 最大值=1.0000
------------------------------------------------------------
2. 测试时间注入残差块 (Residual Block):
   输入特征图形状: torch.Size([4, 16, 32, 32]) (通道数=16)
   输出特征图形状: torch.Size([4, 32, 32, 32]) (通道数=32, 预期: [4, 32, 32, 32])
------------------------------------------------------------
3. 测试自注意力块 (Attention Block):
   输入特征图形状: torch.Size([4, 32, 32, 32])
   输出特征图形状: torch.Size([4, 32, 32, 32]) (预期: [4, 32, 32, 32])
------------------------------------------------------------
4. 测试简易 U-Net 架构:
   输入图像形状: torch.Size([4, 3, 32, 32]) (预期: [4, 3, 32, 32])
   输入时间步形状: torch.Size([4]) (预期: [4])
   网络输出形状: torch.Size([4, 3, 32, 32]) (预期: [4, 3, 32, 32])

   测试成功！简易 U-Net 能够正常前向传播，且输入输出维度完全一致 ✓
============================================================
```

---

## 下一节预告

**Section 05**：完整训练循环与采样 —— 结合前面所有章节的知识，在 MNIST 数据集上从零开始训练我们的去噪 U-Net，并实现反向去噪采样，亲眼见证模型从纯高斯噪声中「凭空」生成手写数字图片！

详见 [`section05_train_sample/README.md`](../section05_train_sample/README.md)。
