# -*- coding: utf-8 -*-
"""
Section 08.6 - 训练迷你 Masked Diffusion LM

在内置英文短句上训练双向 MaskPredictor。
CPU 可用；--fast 约 1 分钟内看到 loss 下降。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from tqdm import tqdm

from data_utils import BUILTIN_SENTENCES, build_vocab, make_dataloader, setup_stdio
from md_model import MaskDiffusionConfig, MaskPredictor, masked_diffusion_loss

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
FIG = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fast", action="store_true", help="更多 epoch，便于 CPU 快速过拟合短句")
    p.add_argument(
        "--no-t-weight",
        action="store_true",
        help="不用 1/t 加权（小语料更稳；默认开启加权以贴近 LLaDA）",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    setup_stdio()
    args = parse_args()
    if args.fast:
        args.epochs = 200
        args.dropout = 0.05
        args.d_model = 192
        args.n_layers = 3
        args.d_ff = 384
        # 教学小语料：默认关掉 1/t，否则 loss 尺度乱跳、难过拟合
        if not args.no_t_weight:
            args.no_t_weight = True

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    print("=" * 60)
    print("Section 08.6 - 训练 Masked Diffusion LM")
    print("=" * 60)
    print(f"device={device}  epochs={args.epochs}  sentences={len(BUILTIN_SENTENCES)}")

    vocab = build_vocab()
    loader = make_dataloader(vocab, batch_size=args.batch_size, shuffle=True)

    cfg = MaskDiffusionConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=vocab.pad_idx,
        mask_idx=vocab.mask_idx,
    )
    model = MaskPredictor(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x0 in loader:
            x0 = x0.to(device)
            loss, _ = masked_diffusion_loss(
                model,
                x0,
                vocab.pad_idx,
                vocab.mask_idx,
                vocab.bos_idx,
                vocab.eos_idx,
                use_t_weight=not args.no_t_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        avg = total / max(n, 1)
        losses.append(avg)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}")

    # 保存
    vocab.save(CKPT / "vocab.json")
    (CKPT / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": cfg.vocab_size,
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers,
                "d_ff": cfg.d_ff,
                "dropout": cfg.dropout,
                "pad_idx": cfg.pad_idx,
                "mask_idx": cfg.mask_idx,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, CKPT / "last.pth")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses, color="#2980b9")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Masked diffusion training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "06_loss_curve.png", dpi=120)
    plt.close(fig)

    print(f"\n已保存: {CKPT / 'last.pth'}")
    print(f"损失曲线: {FIG / '06_loss_curve.png'}")
    print("下一步: python 07_sample.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
