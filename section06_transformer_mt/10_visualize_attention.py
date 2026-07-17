# -*- coding: utf-8 -*-
"""
Section 06.10 - 可视化训练后的 Attention（真正能「看见」模型看了哪里）

前置：先 python 08_train.py --fast

本脚本从 checkpoint 加载模型，对一句英文：
  1. 贪心翻译出中文
  2. 用 Teacher Forcing 的输入再跑一遍 forward_with_attention
  3. 画出 Cross-Attention 热力图（行=中文，列=英文）

默认画「最后一层、各头平均」的对齐矩阵——这是机器翻译里最好读的一张图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_utils import Vocab, tokenize_en
from transformer import (
    TransformerConfig,
    TransformerMT,
    greedy_decode,
    make_pad_mask,
    make_tgt_mask,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
FIG_DIR = HERE / "figures"


def parse_args():
    p = argparse.ArgumentParser(description="Section 06.10 - 可视化 Cross-Attention")
    p.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "last.pth"))
    p.add_argument("--sentence", type=str, default="i love you")
    p.add_argument("--layer", type=int, default=-1, help="Decoder 层索引，-1=最后一层")
    p.add_argument("--head", type=int, default=-1, help="注意力头，-1=各头平均")
    p.add_argument("--show-all-heads", action="store_true", help="额外画每个 head 的子图")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
    ckpt_dir = ckpt_path.parent
    src_vocab = Vocab.load(ckpt_dir / "src_vocab.json")
    tgt_vocab = Vocab.load(ckpt_dir / "tgt_vocab.json")
    if (ckpt_dir / "config.json").exists():
        cfg_dict = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    else:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_dict = blob["config"]
    cfg = TransformerConfig(**cfg_dict)
    model = TransformerMT(cfg).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab


def ids_to_labels(vocab: Vocab, ids: list[int]) -> list[str]:
    return [vocab.id_to_token.get(i, "?") for i in ids]


def plot_heatmap(
    weights,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: Path,
    cmap: str = "Oranges",
):
    """weights: 2D numpy (L_row, L_col)"""
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(col_labels) + 2), max(4, 0.55 * len(row_labels) + 1)))
    im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Key = 英文 (Encoder memory)")
    ax.set_ylabel("Query = 中文 (Decoder)")
    ax.set_title(title)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all_heads(attn_bhqk, row_labels, col_labels, title_prefix: str, path: Path):
    """attn_bhqk: (H, L_tgt, L_src)"""
    H = attn_bhqk.shape[0]
    cols = min(4, H)
    rows = (H + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
    for h in range(H):
        r, c = divmod(h, cols)
        ax = axes[r][c]
        ax.imshow(attn_bhqk[h].numpy(), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_title(f"head {h}", fontsize=10)
    for h in range(H, rows * cols):
        r, c = divmod(h, cols)
        axes[r][c].axis("off")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


@torch.no_grad()
def collect_cross_attention(
    model: TransformerMT,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    sentence: str,
    device: torch.device,
):
    """翻译一句，再取 Cross-Attn。返回 pred 文本、src/tgt 标签、各层 cross attn。"""
    src_ids = src_vocab.encode(tokenize_en(sentence), add_bos_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    out_ids = greedy_decode(
        model,
        src,
        bos_idx=tgt_vocab.bos_idx,
        eos_idx=tgt_vocab.eos_idx,
        pad_idx=src_vocab.pad_idx,
        max_len=32,
    )[0].tolist()
    pred = tgt_vocab.decode(out_ids)

    # Teacher Forcing 风格：tgt_in = 已生成序列去掉最后一个（与训练一致）
    # 若序列以 eos 结尾，用完整 ys 去掉最后一位作为输入
    tgt_in = torch.tensor([out_ids[:-1]], dtype=torch.long, device=device)
    if tgt_in.size(1) == 0:
        tgt_in = torch.tensor([[tgt_vocab.bos_idx]], dtype=torch.long, device=device)

    src_mask = make_pad_mask(src, src_vocab.pad_idx)
    tgt_mask = make_tgt_mask(tgt_in, src_vocab.pad_idx)
    _, attns = model.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    src_labels = ids_to_labels(src_vocab, src_ids)
    # tgt_in 的每个位置预测「下一个字」，热力图行标签用「当前已有输入」更直观：
    # 行 i 对应输入 token[i]，此时模型在预测下一个 → 用 predicted 序列做行标签更贴合翻译对齐
    # 用 tgt_out = out_ids[1:]（被预测的字）当行标签：写「爱」时看英文哪一词
    tgt_out_ids = out_ids[1:]
    if len(tgt_out_ids) > tgt_in.size(1):
        tgt_out_ids = tgt_out_ids[: tgt_in.size(1)]
    elif len(tgt_out_ids) < tgt_in.size(1):
        tgt_out_ids = tgt_out_ids + [tgt_vocab.pad_idx] * (tgt_in.size(1) - len(tgt_out_ids))
    tgt_labels = ids_to_labels(tgt_vocab, tgt_out_ids)

    # cross: list of (1, H, L_tgt, L_src)
    cross_layers = [w[0].cpu() for w in attns["cross"]]  # each (H, Lt, Ls)
    return pred, src_labels, tgt_labels, cross_layers


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)

    print("=" * 60)
    print("Section 06.10 - 可视化 Attention")
    print("=" * 60)

    if not ckpt_path.exists():
        print(f"找不到 checkpoint: {ckpt_path}")
        print("请先运行: python 08_train.py --fast")
        return

    model, src_vocab, tgt_vocab = load_model(ckpt_path, device)
    pred, src_labels, tgt_labels, cross_layers = collect_cross_attention(
        model, src_vocab, tgt_vocab, args.sentence, device
    )

    layer_i = args.layer if args.layer >= 0 else len(cross_layers) - 1
    layer_i = max(0, min(layer_i, len(cross_layers) - 1))
    attn_h = cross_layers[layer_i]  # (H, Lt, Ls)

    if args.head < 0:
        mat = attn_h.mean(dim=0)  # 各头平均
        head_tag = "avg_heads"
        title = f'Cross-Attn layer={layer_i} (avg heads)\n"{args.sentence}" -> "{pred}"'
    else:
        h = min(args.head, attn_h.size(0) - 1)
        mat = attn_h[h]
        head_tag = f"head{h}"
        title = f'Cross-Attn layer={layer_i} head={h}\n"{args.sentence}" -> "{pred}"'

    safe = args.sentence.replace(" ", "_")[:40]
    out_path = FIG_DIR / f"10_cross_attn_{safe}_L{layer_i}_{head_tag}.png"
    plot_heatmap(mat.numpy(), tgt_labels, src_labels, title, out_path)

    print(f"\nEN: {args.sentence}")
    print(f"ZH: {pred}")
    print(f"\n【怎么读这张图】")
    print("  行 = 正在写的中文字（Query）")
    print("  列 = 英文位置（Key / memory）")
    print("  格子越深 = 写这个中文字时，模型越关注该英文词")
    print(f"\n热力图已保存: {out_path}")

    # 打印数字矩阵，方便终端对照
    print("\n【对齐权重矩阵】（所选层/头）")
    header = "       " + "  ".join(f"{c:>6s}" for c in src_labels)
    print(header)
    for i, row_lab in enumerate(tgt_labels):
        row = "  ".join(f"{mat[i, j].item():6.3f}" for j in range(len(src_labels)))
        print(f"  {row_lab:5s} -> [{row}]")

    if args.show_all_heads:
        heads_path = FIG_DIR / f"10_cross_attn_{safe}_L{layer_i}_all_heads.png"
        plot_all_heads(
            attn_h,
            tgt_labels,
            src_labels,
            f'Cross-Attn heads | "{args.sentence}" -> "{pred}"',
            heads_path,
        )
        print(f"各头子图已保存: {heads_path}")

    print("\n提示: 训练中也可自动存图（08_train.py 每 eval-every 会存一张）")
    print("=" * 60)


if __name__ == "__main__":
    main()
