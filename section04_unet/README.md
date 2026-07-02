# Section 04 - U-Net 架构：时间嵌入与残差去噪网络

## 本节目标

- 理解为什么去噪网络 $\varepsilon_\theta(x_t, t)$ 需要输入时间步 $t$
- 掌握正弦时间嵌入（Sinusoidal Position Embedding）的数学原理与实现
- 掌握如何在残差块（Residual Block）中注入时间信息
- 理解自注意力机制（Self-Attention）在扩散模型中的作用
- 学习并组装一个完整的简易去噪 U-Net 网络，验证其输入输出维度一致性

## 本节目录

- 零、为什么去噪网络需要时间步 $t$
- 一、正弦时间嵌入（Sinusoidal Position Embedding）
- 二、时间注入残差块（Residual Block with Time Embedding）
- 三、自注意力机制（Self-Attention）
- 四、简易 U-Net 架构拼装
- 五、维度验证与测试

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

### 3.1 为什么需要注意力机制？

卷积神经网络（CNN）的感受野是局部的。虽然通过堆叠卷积层可以扩大感受野，但在处理全局结构（如图像的对称性、长距离依赖）时，纯 CNN 依然存在局限。

在 U-Net 的低分辨率层（如 $8 \times 8$ 或 $16 \times 16$）引入**自注意力机制**，可以让网络在极低的计算开销下建立全局像素之间的联系，从而显著提高生成图像的全局协调性。

### 3.2 简易单头自注意力实现

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

        # 计算注意力图
        attn = torch.bmm(q, k) * (C ** -0.5)     # [B, N, N]
        attn = torch.softmax(attn, dim=-1)

        # 加权求和并恢复形状
        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return x + self.proj(out)
```

---

## 四、简易 U-Net 架构拼装

### 4.1 整体架构图

我们拼装的 `SimpleUNet` 采用经典的沙漏型结构，并包含**跳跃连接（Skip Connections）**。跳跃连接能够将编码器保留的高频空间细节直接传递给解码器，避免在下采样过程中丢失关键信息。

```
输入 x0 ──> [ Init Conv ] ───────────────────────────────────────────> ( 拼接 ) ──> [ Up2 Res ] ──> [ Out Conv ] ──> 预测噪声
                 │                                                       ▲
             [ Down1 Res ] ─────────────────────────> ( 拼接 ) ──> [ Up1 Res ]   │
                 │                                       ▲               │
             [ Down1 Pool ]                          [ Up1 Unpool ]      │
                 │                                       │               │
             [ Down2 Res ] ──> [ Down2 Pool ] ──> [ Bottleneck ] ────────┘
```

### 4.2 核心前向传播逻辑

在 U-Net 的前向传播中，时间嵌入 `t_emb` 在每一个残差块中都会被传入并注入：

```python
def forward(self, x, t):
    # 1. 计算时间嵌入
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
    h = self.mid_res2(h)
    
    # 4. 解码器 (Upsampling)
    h = self.up1_unpool(h)
    h = torch.cat((h, x2_res), dim=1)           # 拼接跳跃连接 2
    h = self.up1_res(h, t_emb)
    
    h = self.up2_unpool(h)
    h = torch.cat((h, x1_res), dim=1)           # 拼接跳跃连接 1
    h = self.up2_res(h, t_emb)
    
    # 5. 输出投影
    return self.out_conv(h)
```

---

## 五、维度验证与测试

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

**Section 05**：完整训练循环与采样 —— 结合前面所有章节的知识，在 MNIST 数据集上从零开始训练我们的去噪 U-Net，并实现反向去噪采样，亲眼见证模型从纯高斯噪声中“凭空”生成手写数字图片！
