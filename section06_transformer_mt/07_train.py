# -*- coding: utf-8 -*-
"""
Section 06.7 - Transformer 机器翻译训练

前置：建议先跑完 01~06 小节。
Teacher Forcing：输入 tgt[:, :-1]，预测 tgt[:, 1:]
损失：CrossEntropy（ignore <pad>）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import Vocab, make_dataloader, prepare_data
from transformer import TransformerConfig, TransformerMT, make_pad_mask, make_tgt_mask, greedy_decode

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="Section 06.7 - Transformer 英→中翻译训练")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--data", type=str, default="builtin", choices=["builtin", "tsv"])
    p.add_argument("--tsv", type=str, default=None, help="自定义平行语料 TSV 路径")
    p.add_argument("--fast", action="store_true", help="快速演示：120 epoch")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=10, help="每 N epoch 打印翻译样例")
    return p.parse_args()


def plot_loss_curve(train_losses: list[float], val_losses: list[float], path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="train", color="steelblue", lw=1.5)
    ax.plot(val_losses, label="val", color="coral", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Section 06.7 - Transformer MT Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(model, loader, criterion, pad_idx, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        src_mask = make_pad_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)
        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item() * src.size(0)
        n += src.size(0)
    return total / max(n, 1)


@torch.no_grad()
def show_translations(
    model: TransformerMT,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    sentences: list[str],
    device: torch.device,
):
    model.eval()
    print("  --- 翻译样例 ---")
    for s in sentences:
        from data_utils import tokenize_en

        ids = src_vocab.encode(tokenize_en(s), add_bos_eos=True)
        src = torch.tensor([ids], dtype=torch.long, device=device)
        out = greedy_decode(
            model, src,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
            pad_idx=src_vocab.pad_idx,
            max_len=32,
        )
        pred = tgt_vocab.decode(out[0].tolist())
        print(f"  EN: {s}")
        print(f"  ZH: {pred}")
        print()


def save_checkpoint(path: Path, model, optimizer, epoch, cfg_dict, src_vocab, tgt_vocab, loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg_dict,
            "loss": loss,
        },
        path,
    )
    src_vocab.save(path.parent / "src_vocab.json")
    tgt_vocab.save(path.parent / "tgt_vocab.json")
    (path.parent / "config.json").write_text(
        json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train():
    args = parse_args()
    if args.fast:
        args.epochs = 120
        args.eval_every = 20
        args.dropout = 0.05

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 60)
    print("Section 06.7 - Transformer 英→中翻译训练")
    print(f"设备: {device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"模型: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    if args.fast:
        print("模式: --fast")
    print("=" * 60)

    tsv = Path(args.tsv) if args.tsv else None
    train_ds, val_ds, src_vocab, tgt_vocab = prepare_data(
        source=args.data, tsv_path=tsv
    )
    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, args.batch_size, shuffle=False)

    print(f"训练集: {len(train_ds)} 句 | 验证集: {len(val_ds)} 句")
    print(f"源词表: {len(src_vocab)} | 目标词表: {len(tgt_vocab)}")

    cfg = TransformerConfig(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=src_vocab.pad_idx,
    )
    model = TransformerMT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_params:,}")

    cfg_dict = {
        "src_vocab_size": cfg.src_vocab_size,
        "tgt_vocab_size": cfg.tgt_vocab_size,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "n_encoder_layers": cfg.n_encoder_layers,
        "n_decoder_layers": cfg.n_decoder_layers,
        "d_ff": cfg.d_ff,
        "dropout": cfg.dropout,
        "max_len": cfg.max_len,
        "pad_idx": cfg.pad_idx,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)

    demo_sentences = [
        "i love you",
        "good morning",
        "where is the book",
        "i am a student",
        "attention is all you need",
    ]

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_tok = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_pad_mask(src, src_vocab.pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, src_vocab.pad_idx)

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * src.size(0)
            n_tok += src.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(n_tok, 1)
        val_loss = evaluate(model, val_loader, criterion, src_vocab.pad_idx, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            show_translations(model, src_vocab, tgt_vocab, demo_sentences, device)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                CKPT_DIR / "best.pth",
                model, optimizer, epoch, cfg_dict, src_vocab, tgt_vocab, val_loss,
            )
            print(f"  [best] 已保存 (val={val_loss:.4f})")

    save_checkpoint(
        CKPT_DIR / f"transformer_epoch_{args.epochs:03d}.pth",
        model, optimizer, args.epochs, cfg_dict, src_vocab, tgt_vocab, val_losses[-1],
    )
    save_checkpoint(
        CKPT_DIR / "last.pth",
        model, optimizer, args.epochs, cfg_dict, src_vocab, tgt_vocab, val_losses[-1],
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(train_losses, val_losses, FIG_DIR / "loss_curve.png")

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"训练完成！耗时 {elapsed / 60:.1f} 分钟")
    print(f"损失曲线: {FIG_DIR / 'loss_curve.png'}")
    print(f"最佳(val) checkpoint: {CKPT_DIR / 'best.pth'}")
    print(f"最终 checkpoint: {CKPT_DIR / 'last.pth'}  <-- 小语料推理建议用这个")
    print("下一步: python 08_infer.py")
    print("=" * 60)


if __name__ == "__main__":
    train()
