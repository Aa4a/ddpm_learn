# -*- coding: utf-8 -*-
"""
Section 07 - 迷你 Latent Diffusion（教学版 Stable Diffusion）核心模块

三大件：
  1. TinyVAE        —— 像素 ↔ 潜空间
  2. TextEncoder    —— prompt → 上下文向量（简化版 CLIP）
  3. CondUNet       —— 潜空间去噪 + Cross-Attention 文本条件

外加 LatentDDPMSchedule：与 Section 05 相同的加噪/去噪公式，作用在 z 上。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import PAD_ID, UNCOND_ID, VOCAB_SIZE, MAX_TEXT_LEN


# ---------------------------------------------------------------------------
# 时间嵌入（与 Section 04 同源）
# ---------------------------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# ---------------------------------------------------------------------------
# 1. TinyVAE：28x28 → 4x7x7
# ---------------------------------------------------------------------------
class TinyVAE(nn.Module):
    """
    教学用卷积 VAE。
    编码器：28→14→7，输出 mu/logvar（各 latent_ch 通道）
    解码器：7→14→28，重建到 [-1,1]（Tanh）
    """

    def __init__(self, latent_ch: int = 4):
        super().__init__()
        self.latent_ch = latent_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 28→14
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14→7
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.fc_mu = nn.Conv2d(64, latent_ch, 1)
        self.fc_logvar = nn.Conv2d(64, latent_ch, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7→14
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 14→28
            nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """推理 / 扩散训练时常用 mu 作为潜变量（更稳）。"""
        mu, _ = self.encode(x)
        return mu


def vae_loss(recon, x, mu, logvar, kl_weight: float = 1e-4):
    """重建 MSE + 轻量 KL。"""
    recon_loss = F.mse_loss(recon, x)
    # KL(N(mu,σ) || N(0,I))，按元素平均
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl, recon_loss.detach(), kl.detach()


# ---------------------------------------------------------------------------
# 2. TextEncoder（简化版 CLIP Text Encoder）
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """词嵌入 + 可选浅层自注意力，输出 (B, L, ctx_dim)。"""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        ctx_dim: int = 64,
        max_len: int = MAX_TEXT_LEN,
        n_layers: int = 1,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.ctx_dim = ctx_dim
        self.embed = nn.Embedding(vocab_size, ctx_dim, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, ctx_dim))
        layers = []
        for _ in range(n_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=ctx_dim,
                nhead=4,
                dim_feedforward=ctx_dim * 2,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ))
        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(ctx_dim)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        text_ids: (B, L)
        return: context (B, L, ctx_dim)
        """
        x = self.embed(text_ids) + self.pos[:, : text_ids.size(1)]
        # pad mask: True = ignore（PyTorch Transformer 约定）
        key_padding_mask = text_ids == self.pad_id
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 3. Cross-Attention（图像 Q，文本 K/V）—— 对照 Section 06.6
# ---------------------------------------------------------------------------
class CrossAttention2d(nn.Module):
    """
    Q 来自图像特征图，K/V 来自文本 context。
    公式与翻译 Cross-Attn 相同，只是「译文」换成了「图像空间 token」。
    """

    def __init__(self, channels: int, ctx_dim: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Linear(ctx_dim, channels, bias=False)
        self.to_v = nn.Linear(ctx_dim, channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.last_attn: torch.Tensor | None = None  # (B, heads, N, L) 供可视化

    def forward(self, x: torch.Tensor, context: torch.Tensor, return_attn: bool = False):
        """
        x: (B, C, H, W)
        context: (B, L, ctx_dim)
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.to_q(h).view(B, self.n_heads, self.head_dim, H * W).transpose(2, 3)
        # q: (B, heads, N, d)
        k = self.to_k(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        # k,v: (B, heads, L, d)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, N, L)
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach()

        out = torch.matmul(attn, v)  # (B, heads, N, d)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = x + self.proj(out)
        if return_attn:
            return out, attn
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# ---------------------------------------------------------------------------
# 4. CondUNet：潜空间去噪 + Cross-Attn
# ---------------------------------------------------------------------------
class CondUNet(nn.Module):
    """
    输入 z_t (B, latent_ch, 7, 7) + t + text context → 预测噪声 ε。
    结构：下采样 → bottleneck（Self 省略，放 Cross-Attn）→ 上采样。
    """

    def __init__(
        self,
        latent_ch: int = 4,
        time_dim: int = 128,
        base_ch: int = 64,
        ctx_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.time_dim = time_dim
        c1, c2 = base_ch, base_ch * 2

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.init_conv = nn.Conv2d(latent_ch, c1, 3, padding=1)
        self.down1_res = ResidualBlock(c1, c1, time_dim)
        # 7→4（ceil）：用 stride-2 conv，padding=1 → floor((7+2-3)/2)+1 = 4
        self.down1_pool = nn.Conv2d(c1, c1, 3, stride=2, padding=1)

        self.down2_res = ResidualBlock(c1, c2, time_dim)

        self.mid_res1 = ResidualBlock(c2, c2, time_dim)
        self.mid_cross = CrossAttention2d(c2, ctx_dim, n_heads=n_heads)
        self.mid_res2 = ResidualBlock(c2, c2, time_dim)

        self.up1_unpool = nn.Upsample(size=(7, 7), mode="nearest")
        self.up1_res = ResidualBlock(c2 + c1, c1, time_dim)  # cat skip
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, latent_ch, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        return_attn: bool = False,
    ):
        t_emb = self.time_embed(t.float())

        x = self.init_conv(z)
        skip = self.down1_res(x, t_emb)       # (B, c1, 7, 7)
        h = self.down1_pool(skip)             # (B, c1, 4, 4)
        h = self.down2_res(h, t_emb)          # (B, c2, 4, 4)

        h = self.mid_res1(h, t_emb)
        if return_attn:
            h, attn = self.mid_cross(h, context, return_attn=True)
        else:
            h = self.mid_cross(h, context)
            attn = None
        h = self.mid_res2(h, t_emb)

        h = self.up1_unpool(h)                # (B, c2, 7, 7)
        h = torch.cat([h, skip], dim=1)
        h = self.up1_res(h, t_emb)
        out = self.out_conv(h)
        if return_attn:
            return out, attn
        return out


# ---------------------------------------------------------------------------
# 5. 完整 LDM 包装（VAE 冻结 + TextEncoder + CondUNet）
# ---------------------------------------------------------------------------
class MiniLDM(nn.Module):
    def __init__(
        self,
        latent_ch: int = 4,
        time_dim: int = 128,
        base_ch: int = 64,
        ctx_dim: int = 64,
    ):
        super().__init__()
        self.vae = TinyVAE(latent_ch=latent_ch)
        self.text_encoder = TextEncoder(ctx_dim=ctx_dim)
        self.unet = CondUNet(
            latent_ch=latent_ch,
            time_dim=time_dim,
            base_ch=base_ch,
            ctx_dim=ctx_dim,
        )

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_ids)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, text_ids: torch.Tensor):
        context = self.encode_text(text_ids)
        return self.unet(z_t, t, context)


# ---------------------------------------------------------------------------
# 6. Latent DDPM Schedule（公式同 Section 05，作用在 z）
# ---------------------------------------------------------------------------
@dataclass
class DDPMConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class LatentDDPMSchedule:
    def __init__(self, cfg: DDPMConfig, device: torch.device):
        self.cfg = cfg
        T = cfg.num_timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=device), alpha_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = alpha_cumprod_prev
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        shape = (z0.size(0),) + (1,) * (z0.dim() - 1)
        sqrt_ab = self.sqrt_alpha_cumprod[t].view(shape)
        sqrt_omab = self.sqrt_one_minus_alpha_cumprod[t].view(shape)
        return sqrt_ab * z0 + sqrt_omab * noise, noise

    def training_loss(self, model: MiniLDM, z0: torch.Tensor, text_ids: torch.Tensor):
        B = z0.shape[0]
        device = z0.device
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt, _ = self.q_sample(z0, t, noise=noise)
        pred = model(zt, t, text_ids)
        return F.mse_loss(pred, noise)

    def predict_eps(
        self,
        model: MiniLDM,
        zt: torch.Tensor,
        t: int,
        text_ids: torch.Tensor,
        guidance_scale: float = 1.0,
        uncond_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """带 CFG 的噪声预测。"""
        B = zt.shape[0]
        t_batch = torch.full((B,), t, device=zt.device, dtype=torch.long)

        if guidance_scale <= 1.0 or uncond_ids is None:
            return model(zt, t_batch, text_ids)

        # 一次前向：条件与无条件拼 batch
        zt_in = torch.cat([zt, zt], dim=0)
        t_in = torch.cat([t_batch, t_batch], dim=0)
        txt_in = torch.cat([text_ids, uncond_ids], dim=0)
        eps_c, eps_u = model(zt_in, t_in, txt_in).chunk(2, dim=0)
        return eps_u + guidance_scale * (eps_c - eps_u)

    @torch.no_grad()
    def p_sample(
        self,
        model: MiniLDM,
        zt: torch.Tensor,
        t: int,
        text_ids: torch.Tensor,
        guidance_scale: float = 7.5,
        uncond_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eps = self.predict_eps(model, zt, t, text_ids, guidance_scale, uncond_ids)
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_cumprod[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_cumprod[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (zt - (beta_t / sqrt_one_minus_ab) * eps)
        if t == 0:
            return mean
        noise = torch.randn_like(zt)
        return mean + torch.sqrt(self.posterior_variance[t]) * noise

    @torch.no_grad()
    def sample(
        self,
        model: MiniLDM,
        text_ids: torch.Tensor,
        shape: tuple,
        device: torch.device,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        from data_utils import uncond_ids as make_uncond

        B = text_ids.shape[0]
        z = torch.randn(shape, device=device)
        u_ids = make_uncond(B).to(device)
        for t in reversed(range(self.cfg.num_timesteps)):
            z = self.p_sample(model, z, t, text_ids, guidance_scale, u_ids)
        return z
