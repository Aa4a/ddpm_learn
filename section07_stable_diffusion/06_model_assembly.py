# -*- coding: utf-8 -*-
"""
Section 07.6 - 组装完整迷你 LDM + 维度检查

流程：
  x → VAE.encode → z0 → 加噪 z_t → CondUNet(·|text) → ε
  z0 → VAE.decode → x̂
"""

import torch

from data_utils import encode_prompt, setup_stdio
from sd_model import MiniLDM, LatentDDPMSchedule, DDPMConfig


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 07.6 - 组装 MiniLDM")
    print("=" * 60)

    print("""
【整体数据流】

  prompt "three"
       |
       v
  TextEncoder --> context (B, L, D)
       |
  x --VAE Enc--> z0 --q_sample--> z_t --CondUNet(+context)--> eps
       ^                                              |
       +-------- VAE Dec <-- z0 (采样后) <------------+
""")

    device = torch.device("cpu")
    model = MiniLDM(latent_ch=4, base_ch=64, ctx_dim=64).to(device)
    schedule = LatentDDPMSchedule(DDPMConfig(num_timesteps=1000), device)

    B = 2
    x = torch.randn(B, 1, 28, 28)
    text_ids = torch.tensor([encode_prompt("three"), encode_prompt("seven")])

    model.eval()
    with torch.no_grad():
        z0 = model.vae.encode_deterministic(x)
        t = torch.tensor([10, 500])
        zt, noise = schedule.q_sample(z0, t)
        context = model.encode_text(text_ids)
        eps = model.unet(zt, t, context)
        recon = model.vae.decode(z0)
        loss = schedule.training_loss(model, z0, text_ids)

    n_params = sum(p.numel() for p in model.parameters())
    n_unet = sum(p.numel() for p in model.unet.parameters())
    n_vae = sum(p.numel() for p in model.vae.parameters())
    n_txt = sum(p.numel() for p in model.text_encoder.parameters())

    print("【维度追踪】")
    print(f"  x (像素)     : {tuple(x.shape)}")
    print(f"  z0 (潜变量)  : {tuple(z0.shape)}")
    print(f"  z_t          : {tuple(zt.shape)}")
    print(f"  text_ids     : {tuple(text_ids.shape)}")
    print(f"  context      : {tuple(context.shape)}")
    print(f"  eps          : {tuple(eps.shape)}  # 必须与 z_t 同形")
    print(f"  recon        : {tuple(recon.shape)}")
    print(f"  training_loss: {loss.item():.4f}（随机权重，仅验通路）")

    assert eps.shape == zt.shape
    assert recon.shape == x.shape

    print(f"\n【参数量】总 {n_params/1e6:.2f}M  "
          f"(VAE {n_vae/1e6:.2f}M / Text {n_txt/1e6:.2f}M / UNet {n_unet/1e6:.2f}M)")

    print("""
【训练两阶段预告】
  1) 训 VAE：重建图像（冻结后给扩散用）
  2) 训 LDM：在 z 上预测噪声，文本条件 + CFG drop

  下一节：python 07_train.py --fast
""")
    print("=" * 60)
    print("维度检查通过 [OK]")
    print("=" * 60)


if __name__ == "__main__":
    main()
