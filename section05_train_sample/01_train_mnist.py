# -*- coding: utf-8 -*-
"""
Section 05 - 在 MNIST 上训练 DDPM 并采样生成手写数字

训练目标：L_simple = E[||ε - ε_θ(x_t, t)||²]
采样：从 x_T ~ N(0,I) 出发，逐步反向去噪 T 步得到 x_0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from ddpm import DDPMConfig, DDPMSchedule, load_simple_unet

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="MNIST DDPM 训练")
    p.add_argument("--epochs", type=int, default=20, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-samples", type=int, default=64, help="每次采样生成的图片数")
    p.add_argument("--sample-every", type=int, default=5, help="每 N 个 epoch 采样一次")
    p.add_argument("--save-every", type=int, default=5, help="每 N 个 epoch 保存 checkpoint")
    p.add_argument("--fast", action="store_true", help="快速演示：2 epoch + 5000 样本")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动检测")
    return p.parse_args()


def get_dataloader(batch_size: int, fast: bool) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
    ])
    train_set = datasets.MNIST(
        root=str(HERE / "data"),
        train=True,
        download=True,
        transform=transform,
    )
    if fast:
        train_set = Subset(train_set, range(5000))
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


@torch.no_grad()
def save_sample_grid(model, schedule, device, epoch: int, n: int = 64):
    model.eval()
    side = int(n ** 0.5)
    assert side * side == n

    samples = schedule.sample(model, (n, 1, 28, 28), device)
    samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

    fig, axes = plt.subplots(side, side, figsize=(side * 1.2, side * 1.2))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(f"Epoch {epoch} Samples ({n} images, generated from pure noise)", fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"samples_epoch_{epoch:03d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  采样图已保存: {path}")
    model.train()


def plot_loss_curve(losses: list[float], path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="steelblue", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$L_{\rm simple}$ (MSE)")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def train():
    args = parse_args()
    if args.fast:
        args.epochs = 2
        args.sample_every = 1
        args.save_every = 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 60)
    print("Section 05 - MNIST DDPM 训练")
    print(f"设备: {device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    if args.fast:
        print("模式: --fast（5000 样本快速演示）")
    print("=" * 60)

    dataloader = get_dataloader(args.batch_size, args.fast)
    model = load_simple_unet(in_channels=1, time_dim=128).to(device)
    schedule = DDPMSchedule(DDPMConfig(), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    epoch_losses: list[float] = []
    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x0 = batch[0]
            else:
                x0 = batch
            x0 = x0.to(device)

            optimizer.zero_grad()
            loss = schedule.training_loss(model, x0)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch:3d} | 平均损失: {avg_loss:.6f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_sample_grid(model, schedule, device, epoch, n=args.num_samples)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = CKPT_DIR / f"unet_epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)
            print(f"  Checkpoint 已保存: {ckpt_path}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(epoch_losses, FIG_DIR / "loss_curve.png")

    elapsed = time.time() - global_start
    print("-" * 60)
    print(f"训练完成！耗时 {elapsed / 60:.1f} 分钟")
    print(f"损失曲线: {FIG_DIR / 'loss_curve.png'}")
    print(f"最新 checkpoint: {CKPT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
