# -*- coding: utf-8 -*-
"""
Section 07.2 - VAE 与潜空间

演示：
  1. TinyVAE 的压缩比（28×28 → 4×7×7）
  2. 随机初始化时的重建（糊）
  3. 若已有 checkpoint，加载并展示重建对比

运行：
  python 02_vae_latent.py
  python 02_vae_latent.py --checkpoint checkpoints/vae_last.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from data_utils import setup_stdio
from sd_model import TinyVAE

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    setup_stdio()
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 60)
    print("Section 07.2 - VAE 与潜空间")
    print("=" * 60)

    print("""
【为什么需要潜空间？】
  像素空间：28x28x1 = 784 维（MNIST）；真实 SD 常是 512x512x3 ~ 78 万维。
  潜空间：  7x7x4   = 196 维 —— 约 4x 压缩（真实 SD 约 8x 空间下采样）。

  扩散只在 z 上做，U-Net 输入变小，训练/采样都便宜。
""")

    vae = TinyVAE(latent_ch=4).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else HERE / "checkpoints" / "vae_last.pth"
    loaded = False
    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        vae.load_state_dict(state["vae"] if isinstance(state, dict) and "vae" in state else state)
        loaded = True
        print(f"已加载 VAE: {ckpt_path}")
    else:
        print("未找到训练好的 VAE，使用随机初始化（重建会很差，属正常）。")
        print("训完后请跑: python 07_train.py --stage vae --fast")

    # 取几张 MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = datasets.MNIST(root=str(HERE / "data"), train=True, download=True, transform=transform)
    imgs = torch.stack([ds[i][0] for i in range(8)]).to(device)

    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(imgs)
        z = mu  # 确定性编码
        recon = vae.decode(z)

    print(f"\n【维度追踪】")
    print(f"  x     : {tuple(imgs.shape)}   # 像素")
    print(f"  mu/z  : {tuple(z.shape)}   # 潜变量")
    print(f"  recon : {tuple(recon.shape)}")
    print(f"  压缩比: {imgs[0].numel()} -> {z[0].numel()}  "
          f"(约 {imgs[0].numel() / z[0].numel():.1f}x)")

    # 可视化
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
    for i in range(8):
        axes[0, i].imshow(((imgs[i, 0].cpu() + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(((recon[i, 0].cpu() + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("原图", fontsize=11)
    axes[1, 0].set_ylabel("重建", fontsize=11)
    title = "VAE 重建（已训练）" if loaded else "VAE 重建（随机初始化）"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "02_vae_recon.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n重建对比图已保存: {out}")

    print("""
【本节小结】
  - VAE Encoder：x -> z；Decoder：z -> x_hat
  - 扩散训练时通常冻结 VAE，只用 encode 得到 z0
  - 下一节：文本怎么变成向量？-> Text Encoder
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
