# -*- coding: utf-8 -*-
"""
Section 06.8 - Transformer 机器翻译推理

前置：先运行 07_train.py --fast 得到 checkpoint。
从 checkpoint 加载模型，对单句或交互输入做英→中翻译。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_utils import Vocab, tokenize_en
from transformer import TransformerConfig, TransformerMT, greedy_decode

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser(description="Section 06.8 - Transformer 英→中推理")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(CKPT_DIR / "last.pth"),
        help="模型 checkpoint 路径（小语料推荐 last.pth）",
    )
    p.add_argument("--sentence", type=str, default=None, help="单句英文（不加则进入交互模式）")
    p.add_argument("--max-len", type=int, default=32)
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


def translate(
    model: TransformerMT,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    sentence: str,
    device: torch.device,
    max_len: int = 32,
) -> str:
    ids = src_vocab.encode(tokenize_en(sentence), add_bos_eos=True)
    src = torch.tensor([ids], dtype=torch.long, device=device)
    out = greedy_decode(
        model,
        src,
        bos_idx=tgt_vocab.bos_idx,
        eos_idx=tgt_vocab.eos_idx,
        pad_idx=src_vocab.pad_idx,
        max_len=max_len,
    )
    return tgt_vocab.decode(out[0].tolist())


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"找不到 checkpoint: {ckpt_path}")
        print("请先运行: python 07_train.py --fast")
        return

    print("=" * 60)
    print("Section 06.8 - Transformer 推理")
    print(f"加载模型: {ckpt_path}")
    model, src_vocab, tgt_vocab = load_model(ckpt_path, device)
    print(f"设备: {device} | 源词表 {len(src_vocab)} | 目标词表 {len(tgt_vocab)}")
    print("-" * 40)

    demos = [
        "i love you",
        "good morning",
        "where is the school",
        "i am learning machine translation",
        "deep learning changes the world",
    ]

    if args.sentence:
        pred = translate(model, src_vocab, tgt_vocab, args.sentence, device, args.max_len)
        print(f"EN: {args.sentence}")
        print(f"ZH: {pred}")
        return

    print("演示翻译：")
    for s in demos:
        pred = translate(model, src_vocab, tgt_vocab, s, device, args.max_len)
        print(f"  EN: {s}")
        print(f"  ZH: {pred}")
        print()

    print("进入交互模式（输入英文，空行或 quit 退出）：")
    while True:
        try:
            line = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"quit", "exit", "q"}:
            break
        pred = translate(model, src_vocab, tgt_vocab, line, device, args.max_len)
        print(f"ZH> {pred}")


if __name__ == "__main__":
    main()
