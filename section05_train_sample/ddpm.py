# -*- coding: utf-8 -*-
"""
DDPM 核心工具：Beta Schedule、正向加噪、反向采样。
供 Section 05 训练与采样脚本共用。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DDPMConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class DDPMSchedule:
    """线性 Beta Schedule 及预计算 buffer。"""

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
        self.posterior_variance = (
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """
        正向加噪：x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε

        x0: [B, C, H, W], t: [B] int64, 返回 (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        xt = sqrt_ab * x0 + sqrt_omab * noise
        return xt, noise

    def training_loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        """简化损失 L_simple = E[||ε - ε_θ(x_t, t)||²]"""
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise=noise)

        # SimpleUNet 接受 float 时间步
        t_float = t.float()
        noise_pred = model(xt, t_float)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, model, xt: torch.Tensor, t: int) -> torch.Tensor:
        """
        单步反向去噪：x_{t-1} ~ p_θ(x_{t-1} | x_t)

        使用噪声预测形式：
          μ_θ = 1/√α_t ( x_t - β_t/√(1-ᾱ_t) ε_θ )
        """
        B = xt.shape[0]
        device = xt.device
        t_batch = torch.full((B,), t, device=device, dtype=torch.float32)

        eps_pred = model(xt, t_batch)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_cumprod[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_cumprod[t]

        coef = beta_t / sqrt_one_minus_ab
        mean = (1.0 / torch.sqrt(alpha_t)) * (xt - coef * eps_pred)

        if t == 0:
            return mean

        var = self.posterior_variance[t]
        noise = torch.randn_like(xt)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, shape: tuple, device: torch.device) -> torch.Tensor:
        """从纯噪声 x_T 出发，逐步去噪得到 x_0。"""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.cfg.num_timesteps)):
            x = self.p_sample(model, x, t)
        return x


def load_simple_unet(in_channels: int = 1, time_dim: int = 256,
                     channel_1: int = 128, channel_2: int = 256):
    """从 section04 加载 SimpleUNet（避免模块名以数字开头无法直接 import）。"""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "section04_unet" / "01_unet_components.py"
    spec = importlib.util.spec_from_file_location("unet_components", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SimpleUNet(
        in_channels=in_channels, out_channels=in_channels,
        time_dim=time_dim, channel_1=channel_1, channel_2=channel_2,
    )
