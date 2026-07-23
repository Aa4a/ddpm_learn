# -*- coding: utf-8 -*-
"""
Section 08.7 - 迭代 Unmask 采样

从（几乎）全 <mask> 出发，多步预测并保留高置信 token（低置信 remask）。
可选：固定 prompt 前缀（条件生成直觉）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_utils import Vocab, setup_stdio, tokenize
from md_model import MaskDiffusionConfig, MaskPredictor, decode_sample, sample

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(CKPT / "last.pth"))
    p.add_argument("--seq-len", type=int, default=8, help="含 bos/eos 的总长度")
    p.add_argument("--steps", type=int, default=6, help="Unmask 步数")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompt", type=str, default="", help="可选前缀，如 'i love'")
    p.add_argument("--n", type=int, default=5, help="采样条数")
    p.add_argument(
        "--fill",
        type=str,
        default="i <mask> you",
        help="单步填空演示，用字面 <mask>；空字符串跳过",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
    vocab = Vocab.load(CKPT / "vocab.json")
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = blob.get("cfg") or json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
    cfg = MaskDiffusionConfig(**{k: cfg_dict[k] for k in MaskDiffusionConfig.__dataclass_fields__})
    model = MaskPredictor(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, vocab


def encode_with_literal_mask(text: str, vocab: Vocab) -> list[int]:
    """支持字面量 <mask>，例如 'i <mask> you'。"""
    parts = text.lower().strip().split()
    ids = [vocab.bos_idx]
    for p in parts:
        if p == "<mask>":
            ids.append(vocab.mask_idx)
        else:
            ids.append(vocab.token_to_id.get(p, vocab.unk_idx))
    ids.append(vocab.eos_idx)
    return ids


@torch.no_grad()
def one_step_fill(model: MaskPredictor, vocab: Vocab, template: str, device: torch.device):
    ids = encode_with_literal_mask(template, vocab)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)
    pred = logits.argmax(dim=-1)[0]
    out = []
    for i, tok_id in enumerate(ids):
        if tok_id == vocab.mask_idx:
            out.append(vocab.id_to_token[int(pred[i].item())])
        else:
            out.append(vocab.id_to_token[tok_id])
    filled = " ".join(out)
    # 只打印内容词
    content = decode_sample(pred, vocab.id_to_token, skip_special=True)
    # 更好：按位置替换后 decode
    merged = x.clone()
    mask_pos = x == vocab.mask_idx
    merged[mask_pos] = pred.unsqueeze(0)[mask_pos]
    return decode_sample(merged, vocab.id_to_token, skip_special=True), filled


def main():
    setup_stdio()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"找不到 {ckpt}，请先运行: python 06_train.py --fast")
        return

    print("=" * 60)
    print("Section 08.7 - 迭代 Unmask 采样")
    print("=" * 60)

    model, vocab = load_model(ckpt, device)

    if args.fill.strip():
        text, raw = one_step_fill(model, vocab, args.fill, device)
        print(f"\n【单步填空】模板: {args.fill}")
        print(f"  预测: {text}")
        print(f"  raw : {raw}")

    prompt_ids = None
    if args.prompt.strip():
        toks = tokenize(args.prompt)
        prompt_ids = [vocab.bos_idx] + [
            vocab.token_to_id.get(t, vocab.unk_idx) for t in toks
        ]
        print(f"\n条件前缀: {args.prompt!r} → ids {prompt_ids}")

    print(f"\nseq_len={args.seq_len}  steps={args.steps}  temperature={args.temperature}")
    print("【多步 Unmask 采样】")
    for i in range(args.n):
        torch.manual_seed(args.seed + i)
        ids = sample(
            model,
            vocab_size=len(vocab),
            seq_len=args.seq_len,
            mask_idx=vocab.mask_idx,
            pad_idx=vocab.pad_idx,
            bos_idx=vocab.bos_idx,
            eos_idx=vocab.eos_idx,
            steps=args.steps,
            temperature=args.temperature,
            device=device,
            prompt_ids=prompt_ids,
        )
        text = decode_sample(ids, vocab.id_to_token, skip_special=True)
        raw = " ".join(vocab.id_to_token[j] for j in ids[0].tolist())
        print(f"  [{i+1}] {text}")
        print(f"       raw: {raw}")

    print("""
【采样在做什么】
  1. 初始化：内容位全是 <mask>（bos/eos 或 prompt 可见）
  2. 每步：双向模型预测所有 mask 位
  3. 只揭开置信度最高的一小部分，其余继续 mask（remasking）
  4. 重复直到全部揭开

对照连续 DDPM：这里没有 ε，揭开 ≈ 「去噪一步」。
小语料上：先看【单步填空】是否合理；全句生成可能重复训练句式，属正常。
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
