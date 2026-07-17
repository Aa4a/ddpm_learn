# -*- coding: utf-8 -*-
"""
Section 07.5 - Classifier-Free Guidance (CFG)

公式：
  ε̂ = ε_u + s · (ε_c - ε_u)

演示数值上不同 guidance_scale 如何拉大「条件方向」。
"""

import torch

from data_utils import encode_prompt, uncond_ids, setup_stdio
from sd_model import MiniLDM, LatentDDPMSchedule, DDPMConfig


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 07.5 - Classifier-Free Guidance (CFG)")
    print("=" * 60)

    print("""
【动机】
  只喂条件文本训练，模型有时对 prompt「不够听话」。
  CFG 的做法：训练时以一定概率 drop 掉文本（换成 <uncond>）；
  采样时同时算「有条件」和「无条件」噪声，再外推：

    eps_hat = eps_uncond + s * (eps_cond - eps_uncond)

  s = guidance_scale：
    s = 1  -> 等于只用条件（不做引导）
    s > 1  -> 沿着「条件相对无条件」的方向加强
    s 太大 -> 过饱和、失真
  真实 SD 常用 s ~ 7~12；本教学版默认 3~7.5。
""")

    device = torch.device("cpu")
    model = MiniLDM().to(device)
    model.eval()
    schedule = LatentDDPMSchedule(DDPMConfig(num_timesteps=100), device)

    B = 1
    z = torch.randn(B, 4, 7, 7, device=device)
    t = 50
    text_ids = torch.tensor([encode_prompt("three")], device=device)
    u_ids = uncond_ids(B).to(device)

    with torch.no_grad():
        t_batch = torch.full((B,), t, dtype=torch.long)
        eps_c = model(z, t_batch, text_ids)
        eps_u = model(z, t_batch, u_ids)

    print("【数值演示】（随机初始化网络，只看公式行为）")
    print(f"  ||eps_cond||   = {eps_c.norm().item():.4f}")
    print(f"  ||eps_uncond|| = {eps_u.norm().item():.4f}")
    print(f"  ||eps_c-eps_u||= {(eps_c - eps_u).norm().item():.4f}")

    for s in [1.0, 3.0, 7.5, 15.0]:
        eps = eps_u + s * (eps_c - eps_u)
        delta = (eps - eps_c).norm().item()
        print(f"  s={s:5.1f}  ->  ||eps_hat - eps_cond|| = {delta:.4f}  "
              f"(s=1 时应为 0)")

    # 与 schedule.predict_eps 对齐
    with torch.no_grad():
        eps_api = schedule.predict_eps(model, z, t, text_ids, guidance_scale=7.5, uncond_ids=u_ids)
        eps_ref = eps_u + 7.5 * (eps_c - eps_u)
        err = (eps_api - eps_ref).abs().max().item()
    print(f"\n  schedule.predict_eps 与手写公式最大误差: {err:.2e}")

    print("""
【训练侧】（见 data_utils.collate_with_cfg_drop）
  每个 batch 以 ~10% 概率把 text 换成 <uncond>，
  这样同一个 U-Net 既能预测 eps_cond 也能预测 eps_uncond。

【本节小结】
  - CFG = 无条件 + scale * (条件 - 无条件)
  - 下一节：把 VAE + TextEncoder + CondUNet 拼起来验维度
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
