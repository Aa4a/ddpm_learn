# -*- coding: utf-8 -*-
"""
Section 05 - 真实验证集图片 vs 模型生成图片对比
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddpm import DDPMConfig, DDPMSchedule, load_simple_unet

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="验证集 vs 生成图片对比")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def find_latest_checkpoint() -> Path | None:
    if not CKPT_DIR.exists():
        return None
    ckpts = sorted(CKPT_DIR.glob("unet_epoch_*.pth"))
    return ckpts[-1] if ckpts else None


def get_val_images(num_samples: int, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    val_set = datasets.MNIST(
        root=str(HERE / "data"),
        train=False,
        download=True,
        transform=transform,
    )
    val_loader = DataLoader(val_set, batch_size=num_samples, shuffle=True)
    for batch in val_loader:
        x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
        return x0[:num_samples].to(device)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    if ckpt_path is None or not ckpt_path.exists():
        print("未找到 checkpoint")
        raise SystemExit(1)

    print(f"设备: {device}")
    print(f"Checkpoint: {ckpt_path}")

    model = load_simple_unet(in_channels=1, time_dim=128).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = DDPMSchedule(DDPMConfig(), device)
    n = args.num_samples
    side = int(n ** 0.5)

    val_images = get_val_images(n, device)
    generated_images = schedule.sample(model, (n, 1, 28, 28), device)

    val_images = (val_images.clamp(-1, 1) + 1) / 2
    generated_images = (generated_images.clamp(-1, 1) + 1) / 2

    fig, axes = plt.subplots(side, side * 2, figsize=(side * 2.4, side * 1.2))
    for i in range(n):
        row = i // side
        col_val = (i % side) * 2
        col_gen = col_val + 1

        axes[row, col_val].imshow(val_images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, col_val].axis("off")
        if row == 0:
            axes[row, col_val].set_title("Real", fontsize=10)

        axes[row, col_gen].imshow(generated_images[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, col_gen].axis("off")
        if row == 0:
            axes[row, col_gen].set_title("Generated", fontsize=10)

    epoch = ckpt.get("epoch", "?")
    fig.suptitle(f"Real Images vs Generated Images (epoch={epoch})", fontsize=13)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"val_vs_pred_epoch_{epoch}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"对比图已保存: {out_path}")


if __name__ == "__main__":
    main()