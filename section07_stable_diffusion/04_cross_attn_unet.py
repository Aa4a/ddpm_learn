# -*- coding: utf-8 -*-
"""
Section 07.4 - U-Net 中的 Cross-Attention（文生图版）

对照 Section 06.6：
  翻译：Q=译文，K/V=原文 memory
  文生图：Q=图像特征，K/V=文本 context

运行后保存 figures/04_cross_attn_demo.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data_utils import encode_prompt, setup_stdio
from sd_model import CrossAttention2d, TextEncoder, CondUNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def main():
    setup_stdio()
    print("=" * 60)
    print("Section 07.4 - Cross-Attention in CondUNet")
    print("=" * 60)

    print("""
【对照 Section 06.6】

  +-----------------+------------------+------------------+
  |                 | 机器翻译         | 文生图 (SD)      |
  +-----------------+------------------+------------------+
  | Q（谁在问）     | Decoder 译文位置 | 图像空间位置     |
  | K / V           | Encoder memory   | Text embedding  |
  | 对齐矩阵        | (L_tgt, L_src)   | (H*W, L_text)    |
  | 作用            | 译时查原文       | 画时查文字       |
  +-----------------+------------------+------------------+

  公式相同：
    Attention(Q,K,V) = softmax(QK^T / sqrt(d)) * V
""")

    # --- 手工小例子 ---
    print("=" * 60)
    print("【Part 1】手工：2×2 空间 × 文本 [three]")
    print("=" * 60)
    torch.manual_seed(0)
    # 伪造：让某个空间位置更贴近 "three" 的方向
    d, L = 8, 1
    text_labels = ["three"]
    # Q: 4 个空间位置
    Q = torch.randn(4, d)
    K = torch.randn(L, d)
    # 让位置 0 对齐 K
    Q[0] = K[0] + 0.1 * torch.randn(d)
    V = torch.randn(L, d)
    scores = Q @ K.T / (d ** 0.5)
    weights = F.softmax(scores, dim=-1)
    print("  注意力权重 (空间位置 x 文本 token)：")
    print(f"           {text_labels[0]:>8s}")
    for i in range(4):
        print(f"  pos{i}  ->  {weights[i, 0].item():8.3f}")

    # --- 模块 shape ---
    print("\n" + "=" * 60)
    print("【Part 2】CrossAttention2d / CondUNet shape")
    print("=" * 60)
    B, C, H, W, ctx_dim = 2, 64, 4, 4, 64
    x = torch.randn(B, C, H, W)
    enc = TextEncoder(ctx_dim=ctx_dim)
    ids = torch.tensor([encode_prompt("three"), encode_prompt("seven")])
    context = enc(ids)
    cross = CrossAttention2d(C, ctx_dim, n_heads=4)
    out, attn = cross(x, context, return_attn=True)
    print(f"  图像特征 x     : {tuple(x.shape)}")
    print(f"  context        : {tuple(context.shape)}")
    print(f"  CrossAttn out  : {tuple(out.shape)}")
    print(f"  attn weights   : {tuple(attn.shape)}  # (B, heads, N=H*W, L)")

    unet = CondUNet(latent_ch=4, base_ch=64, ctx_dim=ctx_dim)
    z = torch.randn(B, 4, 7, 7)
    t = torch.tensor([100, 500])
    eps, attn_u = unet(z, t, context, return_attn=True)
    print(f"\n  CondUNet:")
    print(f"    z_t   : {tuple(z.shape)}")
    print(f"    eps   : {tuple(eps.shape)}  # 与 z 同形")
    print(f"    attn  : {tuple(attn_u.shape)}")

    # 热力图：平均 head，取 batch0
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w = attn_u[0].mean(0).detach().cpu()  # (N, L)
    N = w.shape[0]
    side = int(N ** 0.5)
    # 每个文本 token 一张空间图
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    im = ax.imshow(w[:, 0].view(side, side).numpy(), cmap="viridis")
    ax.set_title('Cross-Attn map for prompt "three"\n(random init)')
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = FIG_DIR / "04_cross_attn_demo.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n热力图已保存: {path}（随机初始化，训后更有意义）")

    print("""
【本节小结】
  - 文生图 Cross-Attn = 06.6 同款公式，Q/K/V 来源换成「图 / 文」
  - CondUNet 在 bottleneck 插入 mid_cross
  - 下一节：CFG 如何加强「听话程度」
""")
    # (kept ASCII-safe)
    print("=" * 60)


if __name__ == "__main__":
    main()
