# -*- coding: utf-8 -*-
"""
Section 07.7 - 训练迷你 Latent Diffusion

两阶段：
  1) --stage vae  训练 TinyVAE
  2) --stage ldm  冻结 VAE，训练 TextEncoder + CondUNet
  --fast / --stage all  一键串跑（演示用）

示例：
  python 07_train.py --fast
  python 07_train.py --stage vae --epochs 5
  python 07_train.py --stage ldm --epochs 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from tqdm import tqdm

from data_utils import get_dataloader, encode_prompt, DIGIT_NAMES, setup_stdio
from sd_model import (
    MiniLDM,
    TinyVAE,
    LatentDDPMSchedule,
    DDPMConfig,
    vae_loss,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="迷你 LDM 训练")
    p.add_argument("--stage", choices=["vae", "ldm", "all"], default="all")
    p.add_argument("--epochs", type=int, default=None, help="覆盖该阶段默认 epoch")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=200, help="扩散步数（教学可减小）")
    p.add_argument("--kl-weight", type=float, default=1e-4)
    p.add_argument("--cfg-drop", type=float, default=0.1)
    p.add_argument("--fast", action="store_true", help="快速演示：少 epoch + 子集")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--vae-ckpt", type=str, default=None, help="LDM 阶段加载的 VAE")
    return p.parse_args()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def save_vae_recon(vae, loader, device, tag: str):
    vae.eval()
    images, _, _ = next(iter(loader))
    images = images[:8].to(device)
    recon = vae.decode(vae.encode_deterministic(images))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(((images[i, 0].cpu() + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(((recon[i, 0].cpu() + 1) / 2).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    fig.suptitle(f"VAE recon ({tag})")
    fig.tight_layout()
    path = FIG_DIR / f"vae_recon_{tag}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  VAE 重建图: {path}")


@torch.no_grad()
def save_ldm_samples(model, schedule, device, epoch: int, guidance_scale: float = 3.0, fast: bool = False):
    model.eval()
    prompts = DIGIT_NAMES[::2] if fast else DIGIT_NAMES  # fast: 0,2,4,6,8
    ids = torch.tensor([encode_prompt(p) for p in prompts], device=device)
    z = schedule.sample(
        model, ids, shape=(len(prompts), 4, 7, 7),
        device=device, guidance_scale=guidance_scale,
    )
    imgs = model.vae.decode(z)
    imgs = ((imgs.clamp(-1, 1) + 1) / 2).cpu()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = len(prompts)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 1.6, 2.2))
    if cols == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(prompts[i], fontsize=10)
        ax.axis("off")
    fig.suptitle(f"LDM samples epoch {epoch} (CFG={guidance_scale})")
    fig.tight_layout()
    path = FIG_DIR / f"ldm_samples_epoch_{epoch:03d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  LDM 采样图: {path}")


def train_vae(args, device):
    print("=" * 60)
    print("Stage 1: 训练 TinyVAE")
    print("=" * 60)
    epochs = args.epochs if args.epochs is not None else (3 if args.fast else 8)
    batch_size = 256 if args.fast else args.batch_size
    loader = get_dataloader(batch_size, fast=args.fast, drop_prob=0.0, train=True)

    vae = TinyVAE(latent_ch=4).to(device)
    opt = AdamW(vae.parameters(), lr=args.lr)
    history = []

    for epoch in range(1, epochs + 1):
        vae.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"VAE ep{epoch}/{epochs}", leave=False)
        for images, _, _ in pbar:
            images = images.to(device)
            recon, mu, logvar, _ = vae(images)
            loss, rec, kl = vae_loss(recon, images, mu, logvar, kl_weight=args.kl_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * images.size(0)
            n += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec.item():.4f}", kl=f"{kl.item():.4f}")
        avg = total / max(n, 1)
        history.append(avg)
        print(f"  epoch {epoch}: loss={avg:.4f}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = {"vae": vae.state_dict(), "latent_ch": 4}
    torch.save(ckpt, CKPT_DIR / "vae_last.pth")
    save_json({"stage": "vae", "epochs": epochs, "loss": history}, CKPT_DIR / "vae_config.json")
    print(f"  已保存: {CKPT_DIR / 'vae_last.pth'}")
    save_vae_recon(vae, loader, device, tag=f"ep{epochs:03d}")
    return vae


def train_ldm(args, device, vae: TinyVAE | None = None):
    print("=" * 60)
    print("Stage 2: 训练 Latent Diffusion (CondUNet + TextEncoder)")
    print("=" * 60)
    epochs = args.epochs if args.epochs is not None else (5 if args.fast else 15)
    # LDM 采样步数多，fast 用更小 batch 但够用
    batch_size = 128 if args.fast else args.batch_size
    timesteps = 30 if args.fast else args.timesteps
    loader = get_dataloader(batch_size, fast=args.fast, drop_prob=args.cfg_drop, train=True)

    model = MiniLDM(latent_ch=4, base_ch=64, ctx_dim=64).to(device)

    # 加载并冻结 VAE
    vae_path = Path(args.vae_ckpt) if args.vae_ckpt else CKPT_DIR / "vae_last.pth"
    if vae is not None:
        model.vae.load_state_dict(vae.state_dict())
    elif vae_path.is_file():
        state = torch.load(vae_path, map_location=device, weights_only=True)
        model.vae.load_state_dict(state["vae"] if "vae" in state else state)
        print(f"  加载 VAE: {vae_path}")
    else:
        raise FileNotFoundError(f"找不到 VAE checkpoint: {vae_path}，请先 --stage vae")

    for p in model.vae.parameters():
        p.requires_grad = False
    model.vae.eval()

    schedule = LatentDDPMSchedule(DDPMConfig(num_timesteps=timesteps), device)
    params = list(model.unet.parameters()) + list(model.text_encoder.parameters())
    opt = AdamW(params, lr=args.lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.unet.train()
        model.text_encoder.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"LDM ep{epoch}/{epochs}", leave=False)
        for images, text_ids, _ in pbar:
            images = images.to(device)
            text_ids = text_ids.to(device)
            with torch.no_grad():
                z0 = model.vae.encode_deterministic(images)
            loss = schedule.training_loss(model, z0, text_ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * images.size(0)
            n += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg = total / max(n, 1)
        history.append(avg)
        print(f"  epoch {epoch}: loss={avg:.4f}")

        if epoch == epochs or (not args.fast and epoch % max(1, epochs // 2) == 0):
            save_ldm_samples(model, schedule, device, epoch, guidance_scale=3.0, fast=args.fast)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "timesteps": timesteps,
            "latent_ch": 4,
            "base_ch": 64,
            "ctx_dim": 64,
        },
        CKPT_DIR / "ldm_last.pth",
    )
    save_json(
        {
            "stage": "ldm",
            "epochs": epochs,
            "timesteps": timesteps,
            "loss": history,
            "cfg_drop": args.cfg_drop,
        },
        CKPT_DIR / "ldm_config.json",
    )
    print(f"  已保存: {CKPT_DIR / 'ldm_last.pth'}")

    # loss 曲线
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("LDM MSE loss")
    ax.set_title("Latent Diffusion training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ldm_loss_curve.png", dpi=120)
    plt.close(fig)
    return model


def main():
    setup_stdio()
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}, stage={args.stage}, fast={args.fast}")
    t0 = time.time()

    if args.stage in ("vae", "all"):
        # all 时若指定了 epochs，只作用于当前逻辑：VAE 与 LDM 分别用 fast 默认
        vae_args = argparse.Namespace(**vars(args))
        if args.stage == "all":
            vae_args.epochs = None  # 让 fast 默认生效
        vae = train_vae(vae_args, device)
    else:
        vae = None

    if args.stage in ("ldm", "all"):
        ldm_args = argparse.Namespace(**vars(args))
        if args.stage == "all":
            ldm_args.epochs = None
        train_ldm(ldm_args, device, vae=vae if args.stage == "all" else None)

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    print("下一步: python 08_sample.py --prompt three")


if __name__ == "__main__":
    main()
