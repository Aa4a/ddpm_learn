# -*- coding: utf-8 -*-
"""
Section 04 - 正弦时间嵌入 (Sinusoidal Position Embedding) 可视化

展示：
1. 正弦嵌入的数学定义与纯 NumPy 实现
2. 不同频率维度随时间步 t 的变化曲线
3. 全时间步 × 全维度的热力图
4. 不同 t 的嵌入向量对比
5. 相邻时间步的相似度（相对位置关系）
"""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

T = 1000          # DDPM 总时间步
DIM = 128         # 嵌入维度
BASE = 10000.0    # 频率基数（与 Transformer / DDPM 一致）


def sinusoidal_position_embedding(time, dim, base=BASE):
    """
    将标量时间步 t 映射为高维正弦嵌入向量。

    数学公式（与 Transformer 位置编码相同）：
        PE(t, 2i)   = sin( t / base^(2i/dim) )
        PE(t, 2i+1) = cos( t / base^(2i/dim) )

    参数
    ----
    time : array-like, shape (N,)
        时间步，例如 [0, 100, 500, 999]
    dim : int
        输出嵌入维度 d（必须为偶数）
    base : float
        频率基数，默认 10000

    返回
    ----
    pe : ndarray, shape (N, dim)
        正弦位置嵌入矩阵
    """
    time = np.asarray(time, dtype=np.float64)
    half_dim = dim // 2

    # 频率因子: 1 / base^(2i/dim) = exp( -2i * ln(base) / dim )
    # 等价于 torch 实现中的:
    #   exp = log(10000) / (half_dim - 1)
    #   freq = exp(arange(half_dim) * -exp)
    freq_exponent = np.linspace(0, 1, half_dim)
    freqs = np.exp(freq_exponent * (-math.log(base)))  # shape (half_dim,)

    # 外积: time[n] * freqs[i] -> angle[n, i]
    angles = time[:, None] * freqs[None, :]  # (N, half_dim)

    pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)
    return pe


def build_full_timestep_matrix(num_timesteps=T, dim=DIM):
    """构建所有时间步 [0, T-1] 的嵌入矩阵，用于热力图。"""
    t_all = np.arange(num_timesteps, dtype=np.float64)
    return sinusoidal_position_embedding(t_all, dim)


def plot_frequency_curves(save_path):
    """图1：不同频率维度上，嵌入值随 t 的变化曲线。"""
    t = np.arange(T, dtype=np.float64)
    pe = build_full_timestep_matrix()

    # 选取几个有代表性的维度（对应不同频率）
    dims_to_show = [0, 1, 8, 16, 32, 63]  # sin 侧索引；cos 在 dim//2 + i
    dim_labels = []
    for i in dims_to_show:
        dim_labels.append(f"dim {i} (sin, 高频)")
    for i in [0, 16, 32]:
        dim_labels.append(f"dim {DIM // 2 + i} (cos)")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上半：sin 维度
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dims_to_show)))
    for idx, d in enumerate(dims_to_show):
        ax.plot(t, pe[:, d], color=colors[idx], lw=1.2, label=f"dim {d} (sin)")
    ax.set_ylabel("嵌入值")
    ax.set_title("不同频率维度的 sin 分量：i 越小 → 频率越高 → 曲线振荡越快")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    # 下半：cos 维度（选几个）
    ax = axes[1]
    cos_dims = [DIM // 2 + i for i in [0, 16, 32, 63]]
    for idx, d in enumerate(cos_dims):
        ax.plot(t, pe[:, d], color=colors[idx % len(colors)], lw=1.2, label=f"dim {d} (cos)")
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("嵌入值")
    ax.set_title("不同频率维度的 cos 分量")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    fig.suptitle("正弦时间嵌入：各维度随 t 的周期性变化", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(save_path):
    """图2：时间步 × 维度 的热力图。"""
    pe = build_full_timestep_matrix()

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        pe.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=[0, T - 1, 0, DIM - 1],
    )
    ax.axhline(DIM // 2 - 0.5, color="yellow", ls="--", lw=1, alpha=0.8)
    ax.text(T * 0.02, DIM * 0.08, "sin 区域 (dim 0 ~ dim/2-1)", color="yellow", fontsize=10)
    ax.text(T * 0.02, DIM * 0.58, "cos 区域 (dim dim/2 ~ dim-1)", color="yellow", fontsize=10)
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("嵌入维度")
    ax.set_title(f"正弦嵌入热力图：{T} 个时间步 × {DIM} 维")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("嵌入值")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selected_timesteps(save_path):
    """图3：几个典型时间步的嵌入向量对比。"""
    t_show = [0, 50, 200, 500, 800, 999]
    pe = sinusoidal_position_embedding(t_show, DIM)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes = axes.flatten()
    x = np.arange(DIM)

    for ax, ti, vec in zip(axes, t_show, pe):
        colors = ["#2196F3" if i < DIM // 2 else "#FF5722" for i in range(DIM)]
        ax.bar(x, vec, color=colors, width=1.0, alpha=0.85)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"t = {ti}")
        ax.set_xlim(-1, DIM)
        ax.set_xlabel("维度")
        if ax is axes[0] or ax is axes[3]:
            ax.set_ylabel("嵌入值")

    fig.suptitle(
        "不同时间步的嵌入向量（蓝=sin, 橙=cos）—— 每个 t 都有唯一“指纹”",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_similarity(save_path):
    """图4：相邻时间步的余弦相似度，展示相对位置可学习性。"""
    pe = build_full_timestep_matrix()
    # 归一化
    norms = np.linalg.norm(pe, axis=1, keepdims=True)
    pe_norm = pe / (norms + 1e-8)

    # 相邻步相似度
    sim_adjacent = np.sum(pe_norm[:-1] * pe_norm[1:], axis=1)

    # 与 t=0 的相似度
    sim_to_zero = pe_norm @ pe_norm[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(np.arange(1, T), sim_adjacent, color="#4CAF50", lw=1)
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("cos(PE(t), PE(t+1))")
    ax.set_title("相邻时间步嵌入的余弦相似度（非常接近 → 平滑过渡）")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.95, 1.001)

    ax = axes[1]
    ax.plot(np.arange(T), sim_to_zero, color="#9C27B0", lw=1)
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("cos(PE(t), PE(0))")
    ax.set_title("各时间步与 t=0 的相似度（随 t 增大而单调变化）")
    ax.grid(True, alpha=0.3)

    fig.suptitle("正弦嵌入的相对位置性质：相近 t → 相近向量", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_formula_diagram(save_path):
    """图5：公式示意 + 频率随 i 衰减。"""
    half_dim = DIM // 2
    i = np.arange(half_dim)
    # 波长 ≈ 2π × base^(2i/dim)，即完成一个周期所需的时间步跨度
    wavelengths = 2 * np.pi * np.exp(i / (half_dim - 1) * math.log(BASE))

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    ax = fig.add_subplot(gs[0])
    ax.semilogy(i, wavelengths, "o-", color="#E91E63", ms=3, lw=1.5)
    ax.set_xlabel("维度索引 i")
    ax.set_ylabel("近似波长 (t 的单位)")
    ax.set_title("频率随 i 指数衰减：i 大 → 波长长 → 变化慢")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5)
    ax.axvline(half_dim - 1, color="gray", lw=0.5)

    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    formula_text = (
        r"正弦时间嵌入公式" + "\n\n"
        r"$PE(t,\,2i)   = \sin\!\left(\dfrac{t}{10000^{2i/d}}\right)$" + "\n\n"
        r"$PE(t,\,2i+1) = \cos\!\left(\dfrac{t}{10000^{2i/d}}\right)$" + "\n\n"
        f"其中 d = {DIM},  t ∈ [0, {T - 1}]\n\n"
        "实现步骤：\n"
        "1. 计算频率 freqs[i] = 10000^(-2i/d)\n"
        "2. 角度 angles = t × freqs\n"
        "3. 拼接 [sin(angles), cos(angles)]"
    )
    ax.text(
        0.05, 0.95, formula_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.9),
        family="monospace",
    )

    fig.suptitle("正弦嵌入：多尺度频率编码时间步", fontsize=14)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 60)
    print("正弦时间嵌入可视化")
    print("=" * 60)

    # 演示函数调用
    t_demo = np.array([0, 100, 500, 999])
    emb = sinusoidal_position_embedding(t_demo, dim=128)
    print(f"输入 t: {t_demo.tolist()}")
    print(f"输出形状: {emb.shape}  (预期 [4, 128])")
    print(f"t=0  前 8 维: {np.round(emb[0, :8], 4).tolist()}")
    print(f"t=500 前 8 维: {np.round(emb[2, :8], 4).tolist()}")
    print("-" * 60)

    paths = {
        "01_frequency_curves.png": plot_frequency_curves,
        "02_heatmap.png": plot_heatmap,
        "03_selected_timesteps.png": plot_selected_timesteps,
        "04_similarity.png": plot_similarity,
        "05_formula_diagram.png": plot_formula_diagram,
    }

    for name, fn in paths.items():
        path = os.path.join(OUTPUT_DIR, name)
        fn(path)
        print(f"已保存: {path}")

    print("=" * 60)
    print("全部可视化完成！")
