# -*- coding: utf-8 -*-
"""
Section 07.9 - 可视化 Cross-Attention（潜空间 × 文本）

示例：
  python 09_visualize_cross_attn.py --prompt three
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_utils import encode_prompt, decode_ids, setup_stdio
from sd_model import MiniLDM, LatentDDPMSchedule, DDPMConfig

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="three")
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-vis", type=int, default=None, help="在哪个时间步抓 attn（默认中间）")
    return p.parse_args()


def main():
    setup_stdio()
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(args.checkpoint) if args.checkpoint else CKPT_DIR / "ldm_last.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"找不到 {ckpt}，请先: python 07_train.py --fast")

    print("=" * 60)
    print("Section 07.9 - 可视化 Cross-Attention")
    print("=" * 60)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = MiniLDM(
        latent_ch=state.get("latent_ch", 4),
        base_ch=state.get("base_ch", 64),
        ctx_dim=state.get("ctx_dim", 64),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    timesteps = int(state.get("timesteps", 200))
    schedule = LatentDDPMSchedule(DDPMConfig(num_timesteps=timesteps), device)

    torch.manual_seed(args.seed)
    text_ids = torch.tensor([encode_prompt(args.prompt)], device=device)
    from data_utils import uncond_ids
    u_ids = uncond_ids(1).to(device)

    t_vis = args.t_vis if args.t_vis is not None else timesteps // 2
    z = torch.randn(1, 4, 7, 7, device=device)
    attn_map = None

    with torch.no_grad():
        for t in reversed(range(timesteps)):
            if t == t_vis:
                # 用条件分支抓 attn（不做 CFG 拼接，便于看单一 context）
                t_batch = torch.full((1,), t, device=device, dtype=torch.long)
                context = model.encode_text(text_ids)
                _, attn = model.unet(z, t_batch, context, return_attn=True)
                attn_map = attn[0].mean(0).cpu()  # (N, L)
            z = schedule.p_sample(
                model, z, t, text_ids,
                guidance_scale=args.guidance_scale,
                uncond_ids=u_ids,
            )
        img = model.vae.decode(z)
        img = ((img.clamp(-1, 1) + 1) / 2).cpu()[0, 0]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f'sample "{args.prompt}"')
    axes[0].axis("off")

    if attn_map is not None:
        N, L = attn_map.shape
        side = int(round(N ** 0.5))
        # 对每个文本 token 画空间注意力；教学版 L=1
        heat = attn_map[:, 0].view(side, side).numpy()
        im = axes[1].imshow(heat, cmap="magma")
        axes[1].set_title(f"Cross-Attn @ t={t_vis}\ntoken={decode_ids(text_ids[0])!r}")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    else:
        axes[1].text(0.5, 0.5, "no attn", ha="center")
        axes[1].axis("off")

    fig.tight_layout()
    safe = args.prompt.replace(" ", "_")
    out = FIG_DIR / f"09_cross_attn_{safe}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")
    print("""
【怎么读图】
  左：最终生成的数字图
  右：bottleneck 上各空间位置对文本 token 的注意力（头平均）
  训好后，空间能量往往集中在笔画区域附近（弱对齐，别期望完美）。
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
