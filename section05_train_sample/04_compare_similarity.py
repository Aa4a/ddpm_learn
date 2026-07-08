# -*- coding: utf-8 -*-
"""
比较真实图片和生成图片的相似度
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddpm import DDPMConfig, DDPMSchedule, load_simple_unet

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description="Compare real vs generated images")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=100)
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

    model = load_simple_unet(in_channels=1, time_dim=256, channel_1=128, channel_2=256).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = DDPMSchedule(DDPMConfig(), device)
    n = args.num_samples

    val_images = get_val_images(n, device)
    generated_images = schedule.sample(model, (n, 1, 28, 28), device)

    val_images = (val_images.clamp(-1, 1) + 1) / 2
    generated_images = (generated_images.clamp(-1, 1) + 1) / 2

    mse = F.mse_loss(val_images, generated_images).item()
    mae = F.l1_loss(val_images, generated_images).item()

    val_mean = val_images.mean().item()
    val_std = val_images.std().item()
    gen_mean = generated_images.mean().item()
    gen_std = generated_images.std().item()

    print("=" * 60)
    print(f"真实图片 vs 生成图片对比分析（n={n}）")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)
    print(f"均值 (Mean):")
    print(f"  真实图片: {val_mean:.4f}")
    print(f"  生成图片: {gen_mean:.4f}")
    print(f"标准差 (Std):")
    print(f"  真实图片: {val_std:.4f}")
    print(f"  生成图片: {gen_std:.4f}")
    print(f"\n相似度指标:")
    print(f"  MSE (均方误差): {mse:.6f}")
    print(f"  MAE (平均绝对误差): {mae:.6f}")
    print("=" * 60)
    print("\n说明:")
    print("  - MSE/MAE 越小表示越相似")
    print("  - MSE < 0.1 表示生成质量较好")
    print("  - MSE < 0.05 表示生成质量很好")
    print("=" * 60)


if __name__ == "__main__":
    main()