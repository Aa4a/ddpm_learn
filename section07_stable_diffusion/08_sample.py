# -*- coding: utf-8 -*-
"""
Section 07.8 - 按 prompt 采样（文生图推理）

示例：
  python 08_sample.py --prompt three
  python 08_sample.py --prompt seven --guidance-scale 5
  python 08_sample.py --all-digits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_utils import encode_prompt, DIGIT_NAMES, setup_stdio
from sd_model import MiniLDM, LatentDDPMSchedule, DDPMConfig

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="three")
    p.add_argument("--all-digits", action="store_true", help="生成 0-9 各一张")
    p.add_argument("--num-samples", type=int, default=8, help="同一 prompt 生成张数")
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> tuple[MiniLDM, int]:
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = MiniLDM(
        latent_ch=state.get("latent_ch", 4),
        base_ch=state.get("base_ch", 64),
        ctx_dim=state.get("ctx_dim", 64),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    timesteps = int(state.get("timesteps", 200))
    return model, timesteps


@torch.no_grad()
def sample_prompts(model, schedule, prompts, device, guidance_scale, seed):
    torch.manual_seed(seed)
    ids = torch.tensor([encode_prompt(p) for p in prompts], device=device)
    z = schedule.sample(
        model, ids,
        shape=(len(prompts), 4, 7, 7),
        device=device,
        guidance_scale=guidance_scale,
    )
    imgs = model.vae.decode(z)
    return ((imgs.clamp(-1, 1) + 1) / 2).cpu()


def main():
    setup_stdio()
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = Path(args.checkpoint) if args.checkpoint else CKPT_DIR / "ldm_last.pth"
    if not ckpt.is_file():
        raise FileNotFoundError(f"找不到 {ckpt}，请先: python 07_train.py --fast")

    print("=" * 60)
    print("Section 07.8 - 文生图采样")
    print("=" * 60)
    print(f"checkpoint: {ckpt}")
    print(f"guidance_scale: {args.guidance_scale}")

    model, timesteps = load_model(ckpt, device)
    schedule = LatentDDPMSchedule(DDPMConfig(num_timesteps=timesteps), device)

    if args.all_digits:
        prompts = list(DIGIT_NAMES)
        imgs = sample_prompts(model, schedule, prompts, device, args.guidance_scale, args.seed)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(prompts[i])
            ax.axis("off")
        fig.suptitle(f"All digits (CFG={args.guidance_scale}, T={timesteps})")
        fig.tight_layout()
        out = FIG_DIR / "08_sample_all_digits.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {out}")
    else:
        prompts = [args.prompt] * args.num_samples
        print(f"prompt: {args.prompt!r} × {args.num_samples}")
        imgs = sample_prompts(model, schedule, prompts, device, args.guidance_scale, args.seed)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        cols = min(4, args.num_samples)
        rows = (args.num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = [axes] if rows * cols == 1 else list(axes.flat)
        for i, ax in enumerate(axes):
            if i < args.num_samples:
                ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
                ax.set_title(args.prompt)
            ax.axis("off")
        fig.suptitle(f'prompt="{args.prompt}" CFG={args.guidance_scale}')
        fig.tight_layout()
        safe = args.prompt.replace(" ", "_")
        out = FIG_DIR / f"08_sample_{safe}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {out}")

    print("""
【推理流程回顾】
  1. TextEncoder(prompt) -> context
  2. z_T ~ N(0,I)
  3. for t=T-1..0: CFG 去噪（条件 + 无条件）
  4. VAE.decode(z0) -> 图像
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
