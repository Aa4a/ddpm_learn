# -*- coding: utf-8 -*-
"""
Section 05 - 从 checkpoint 加载模型并采样生成 MNIST 风格数字
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ddpm import DDPMConfig, DDPMSchedule, load_simple_unet

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="DDPM 采样")
    p.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint 路径")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--output", type=str, default=None, help="输出图片路径")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def find_latest_checkpoint() -> Path | None:
    if not CKPT_DIR.exists():
        return None
    ckpts = sorted(CKPT_DIR.glob("unet_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print("未找到 checkpoint。请先运行: python 01_train_mnist.py")
        raise SystemExit(1)

    print("=" * 60)
    print("Section 05 - DDPM 采样")
    print(f"设备: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    model = load_simple_unet(in_channels=1, time_dim=128).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = DDPMSchedule(DDPMConfig(), device)
    n = args.num_samples
    side = int(n ** 0.5)

    print(f"从纯噪声采样 {n} 张图片（{schedule.cfg.num_timesteps} 步反向去噪）...")
    samples = schedule.sample(model, (n, 1, 28, 28), device)
    samples = (samples.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(side, side, figsize=(side * 1.2, side * 1.2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    epoch = ckpt.get("epoch", "?")
    fig.suptitle(f"DDPM Samples (checkpoint epoch={epoch})", fontsize=13)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else FIG_DIR / "samples_latest.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"采样完成，图片已保存: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
