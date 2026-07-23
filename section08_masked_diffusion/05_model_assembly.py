# -*- coding: utf-8 -*-
"""
Section 08.5 - 组装 MaskPredictor + 维度检查

确认：含 <mask> 的输入 → 双向 Encoder → 每个位置 V 维 logits。
"""

import torch

from data_utils import build_vocab, make_dataloader, setup_stdio
from md_model import MaskDiffusionConfig, MaskPredictor, masked_diffusion_loss, q_sample


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 08.5 - 模型组装与维度检查")
    print("=" * 60)

    vocab = build_vocab()
    cfg = MaskDiffusionConfig(
        vocab_size=len(vocab),
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        pad_idx=vocab.pad_idx,
        mask_idx=vocab.mask_idx,
    )
    model = MaskPredictor(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"词表大小 V = {len(vocab)}")
    print(f"参数量    ≈ {n_params:,}")

    loader = make_dataloader(vocab, batch_size=4, shuffle=False)
    x0 = next(iter(loader))
    print(f"\nbatch x0 shape = {tuple(x0.shape)}  (B, L)")

    xt, mask = q_sample(
        x0,
        mask_ratio=0.5,
        mask_idx=vocab.mask_idx,
        pad_idx=vocab.pad_idx,
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
    )
    logits = model(xt)
    print(f"xt shape       = {tuple(xt.shape)}")
    print(f"logits shape   = {tuple(logits.shape)}  期望 (B, L, V={len(vocab)})")
    assert logits.shape == (x0.shape[0], x0.shape[1], len(vocab))

    loss, stats = masked_diffusion_loss(
        model, x0, vocab.pad_idx, vocab.mask_idx, vocab.bos_idx, vocab.eos_idx
    )
    print(f"\n试算 loss = {loss.item():.4f}")
    print(f"stats     = {stats}")

    print("""
【数据流】
  x0 (B,L)
    --q_sample(t)-->  xt（部分 <mask>）
    --MaskPredictor--> logits (B,L,V)
    --只在 mask 位 CE--> loss

无 U-Net、无 VAE、无 Causal Mask。
下一节真正训练。
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
