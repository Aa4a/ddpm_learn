# -*- coding: utf-8 -*-
"""
Section 04 - U-Net 架构与时间嵌入实现
1. 实现正弦时间嵌入 (Sinusoidal Position Embedding)
2. 实现带有时间注入的残差块 (Residual Block with Time Embedding)
3. 实现空间自注意力块 (Self-Attention Block)
4. 组装一个简易的去噪 U-Net 网络
5. 进行前向传播测试，验证输入输出维度一致性
"""

import math
import torch
import torch.nn as nn

# =====================================================================
# 1. 正弦时间嵌入 (Sinusoidal Position Embedding)
# =====================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """
    将标量时间步 t (形状为 [B]) 转换为高维向量 (形状为 [B, dim])。
    使用 Transformer 中经典的正弦和余弦位置编码公式：
    PE(t, 2i)   = sin(t / 10000^(2i/dim))
    PE(t, 2i+1) = cos(t / 10000^(2i/dim))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # 计算指数衰减因子: 10000^(-2i/dim)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # time [B] -> [B, 1], embeddings [half_dim] -> [1, half_dim]
        # 外积得到 [B, half_dim]
        embeddings = time[:, None] * embeddings[None, :]
        # 拼接 sin 和 cos 得到 [B, dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# =====================================================================
# 2. 带有时间注入的残差块 (Residual Block with Time Embedding)
# =====================================================================
class ResidualBlock(nn.Module):
    """
    残差块是 U-Net 的基本构建单元。
    在去噪任务中，网络需要知道当前处于哪一个时间步 t，因此我们需要将时间嵌入向量注入到特征图里。
    注入方式：将时间嵌入通过一个 MLP 投影到与特征图通道数一致的维度，然后加在特征图上。
    """
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        # 时间嵌入的投影层：SiLU 激活 + 线性层
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # 第一层卷积：卷积 -> 组归一化 (GroupNorm) -> SiLU 激活
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )
        
        # 第二层卷积：卷积 -> 组归一化 (GroupNorm) -> SiLU 激活
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU()
        )
        
        # 如果输入输出通道不一致，使用 1x1 卷积调整残差连接的通道数
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        # x: [B, in_channels, H, W]
        # t: [B, time_channels]
        
        # 1. 特征图第一层卷积
        h = self.conv1(x)  # [B, out_channels, H, W]
        
        # 2. 投影时间嵌入并增加维度以匹配特征图 [B, out_channels] -> [B, out_channels, 1, 1]
        time_emb = self.time_mlp(t)[:, :, None, None]
        
        # 3. 将时间信息加到特征图上
        h = h + time_emb
        
        # 4. 特征图第二层卷积
        h = self.conv2(h)  # [B, out_channels, H, W]
        
        # 5. 残差连接
        return h + self.shortcut(x)


# =====================================================================
# 3. 空间自注意力块 (Self-Attention Block)
# =====================================================================
class AttentionBlock(nn.Module):
    """
    在低分辨率特征图上使用自注意力机制，能够帮助模型捕捉全局上下文信息。
    这里实现的是一个简易的单头自注意力块。
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gn = nn.GroupNorm(num_groups=8, num_channels=channels)
        # 一步计算出 Q, K, V 投影
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 归一化
        h = self.gn(x)
        # 计算 Q, K, V
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)  # 拆分成三个 [B, C, H, W]
        
        # 展平空间维度 H, W -> N
        q = q.view(B, C, H * W).transpose(1, 2)  # [B, N, C]
        k = k.view(B, C, H * W)                  # [B, C, N]
        v = v.view(B, C, H * W).transpose(1, 2)  # [B, N, C]

        # 计算注意力权重矩阵：Q 和 K 的内积
        # [B, N, C] * [B, C, N] -> [B, N, N]
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = torch.softmax(attn, dim=-1)

        # 加权求和：注意力权重与 V 的乘积
        # [B, N, N] * [B, N, C] -> [B, N, C]
        out = torch.bmm(attn, v)
        # 恢复形状为 [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # 投影并加上残差
        return x + self.proj(out)


# =====================================================================
# 4. 简易去噪 U-Net 网络 (Simple Denoising U-Net)
# =====================================================================
class SimpleUNet(nn.Module):
    """
    一个用于演示的简易去噪 U-Net 网络。
    结构包含：
    - 编码器 (Downsampling Path)：两层下采样
    - 中间层 (Bottleneck)：残差块 + 注意力块 + 残差块
    - 解码器 (Upsampling Path)：两层上采样 + 跳跃连接 (Skip Connections)
    - 输出层：卷积映射回图像通道数
    """
    def __init__(self, in_channels=3, out_channels=3, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        
        # 1. 时间嵌入层
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 2. 编码器 (Downsampling)
        self.init_conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        
        self.down1_res = ResidualBlock(16, 16, time_dim)
        self.down1_pool = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1) # 尺寸减半
        
        self.down2_res = ResidualBlock(16, 32, time_dim)
        self.down2_pool = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1) # 尺寸减半
        
        # 3. 中间层 (Bottleneck)
        self.mid_res1 = ResidualBlock(32, 32, time_dim)
        self.mid_attn = AttentionBlock(32)
        self.mid_res2 = ResidualBlock(32, 32, time_dim)
        
        # 4. 解码器 (Upsampling)
        # 上采样使用转置卷积 (ConvTranspose2d)
        self.up1_unpool = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        # 跳跃连接拼接后通道数：32 (来自上采样) + 32 (来自 down2_res) = 64
        self.up1_res = ResidualBlock(64, 16, time_dim)
        
        self.up2_unpool = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        # 跳跃连接拼接后通道数：16 (来自上采样) + 16 (来自 down1_res) = 32
        self.up2_res = ResidualBlock(32, 16, time_dim)
        
        # 5. 输出投影层
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x, t):
        # x: [B, in_channels, H, W]
        # t: [B] (标量时间步)
        
        # 1. 计算时间嵌入
        t_emb = self.time_embed(t)  # [B, time_dim]
        
        # 2. 编码器前向传播
        x1 = self.init_conv(x)      # [B, 16, H, W]
        x1_res = self.down1_res(x1, t_emb)  # [B, 16, H, W]  <-- 准备跳跃连接 1
        x1_pool = self.down1_pool(x1_res)   # [B, 16, H/2, W/2]
        
        x2_res = self.down2_res(x1_pool, t_emb) # [B, 32, H/2, W/2] <-- 准备跳跃连接 2
        x2_pool = self.down2_pool(x2_res)       # [B, 32, H/4, W/4]
        
        # 3. 中间层前向传播
        h = self.mid_res1(x2_pool, t_emb)   # [B, 32, H/4, W/4]
        h = self.mid_attn(h)                # [B, 32, H/4, W/4]
        h = self.mid_res2(h, t_emb)          # [B, 32, H/4, W/4]
        
        # 4. 解码器前向传播
        h = self.up1_unpool(h)              # [B, 32, H/2, W/2]
        # 拼接跳跃连接 2
        h = torch.cat((h, x2_res), dim=1)   # [B, 64, H/2, W/2]
        h = self.up1_res(h, t_emb)          # [B, 16, H/2, W/2]
        
        h = self.up2_unpool(h)              # [B, 16, H, W]
        # 拼接跳跃连接 1
        h = torch.cat((h, x1_res), dim=1)   # [B, 32, H, W]
        h = self.up2_res(h, t_emb)          # [B, 16, H, W]
        
        # 5. 输出投影
        out = self.out_conv(h)              # [B, out_channels, H, W]
        return out


# =====================================================================
# 5. 前向传播测试
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DDPM U-Net 组件与架构测试")
    print("=" * 60)
    
    # 模拟输入参数
    batch_size = 4
    channels = 3
    height, width = 32, 32
    
    # 1. 测试时间嵌入
    print("1. 测试时间嵌入 (Sinusoidal Position Embedding):")
    t_test = torch.tensor([0, 100, 500, 999], dtype=torch.float32)
    time_emb_layer = SinusoidalPositionEmbeddings(dim=128)
    t_emb = time_emb_layer(t_test)
    print(f"   输入时间步 t: {t_test.tolist()}")
    print(f"   输出嵌入形状: {t_emb.shape} (预期: [4, 128])")
    print(f"   数值范围: 最小值={t_emb.min().item():.4f}, 最大值={t_emb.max().item():.4f}")
    print("-" * 60)
    
    # 2. 测试残差块
    print("2. 测试时间注入残差块 (Residual Block):")
    x_test = torch.randn(batch_size, 16, height, width)
    res_block = ResidualBlock(in_channels=16, out_channels=32, time_channels=128)
    res_out = res_block(x_test, t_emb)
    print(f"   输入特征图形状: {x_test.shape} (通道数=16)")
    print(f"   输出特征图形状: {res_out.shape} (通道数=32, 预期: [4, 32, 32, 32])")
    print("-" * 60)
    
    # 3. 测试注意力块
    print("3. 测试自注意力块 (Attention Block):")
    attn_block = AttentionBlock(channels=32)
    attn_out = attn_block(res_out)
    print(f"   输入特征图形状: {res_out.shape}")
    print(f"   输出特征图形状: {attn_out.shape} (预期: [4, 32, 32, 32])")
    print("-" * 60)
    
    # 4. 测试完整简易 U-Net
    print("4. 测试简易 U-Net 架构:")
    img_input = torch.randn(batch_size, channels, height, width)
    unet = SimpleUNet(in_channels=channels, out_channels=channels, time_dim=128)
    
    # 前向传播预测噪声
    predicted_noise = unet(img_input, t_test)
    print(f"   输入图像形状: {img_input.shape} (预期: [4, 3, 32, 32])")
    print(f"   输入时间步形状: {t_test.shape} (预期: [4])")
    print(f"   网络输出形状: {predicted_noise.shape} (预期: [4, 3, 32, 32])")
    
    # 验证输入输出形状是否完全一致
    assert img_input.shape == predicted_noise.shape, "错误：U-Net 输入与输出形状不一致！"
    print("\n   测试成功！简易 U-Net 能够正常前向传播，且输入输出维度完全一致 [OK]")
    print("=" * 60)
