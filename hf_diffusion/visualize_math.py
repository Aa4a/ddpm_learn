"""
HF DDPM 数学可视化（纯 NumPy / Matplotlib，无需 diffusers / GPU）

对照 hf_diffusion_train.py 中的 DDPMScheduler 默认行为，手写等价数学实现。
运行: python visualize_math.py
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent / "assets"
OUT_DIR.mkdir(exist_ok=True)

T = 1000
BETA_START = 1e-4
BETA_END = 0.02
np.random.seed(42)


# ---------------------------------------------------------------------------
# 等价于 diffusers.DDPMScheduler(num_train_timesteps=1000) 的默认线性 schedule
# ---------------------------------------------------------------------------
def make_schedule(num_train_timesteps=T, beta_start=BETA_START, beta_end=BETA_END):
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
    }


def add_noise(original_samples, noise, timesteps, alphas_cumprod):
    """等价于 noise_scheduler.add_noise(original_samples, noise, timesteps)"""
    ab = alphas_cumprod[timesteps]
    sqrt_ab = np.sqrt(ab)
    sqrt_1mab = np.sqrt(1.0 - ab)
    # broadcast: timesteps 可以是标量或 per-sample 索引
    shape = list(original_samples.shape)
    if np.ndim(timesteps) == 0:
        t = int(timesteps)
        return sqrt_ab * original_samples + sqrt_1mab * noise
    # batch: timesteps shape (B,), samples shape (B, C, H, W)
    while sqrt_ab.ndim < original_samples.ndim:
        sqrt_ab = sqrt_ab[:, None]
        sqrt_1mab = sqrt_1mab[:, None]
    return sqrt_ab * original_samples + sqrt_1mab * noise


def forward_step(x_prev, t, alphas, betas):
    eps = np.random.randn(*x_prev.shape)
    return np.sqrt(alphas[t]) * x_prev + np.sqrt(betas[t]) * eps


def posterior_params(sched):
    betas = sched["betas"]
    alphas = sched["alphas"]
    ab = sched["alphas_cumprod"]
    ab_prev = sched["alphas_cumprod_prev"]
    # t = 1..T-1 对应数组索引 1..T-1（HF 用 0-index: timestep 0 对应 beta[0]）
    w1 = np.sqrt(ab_prev[1:]) * betas[1:] / (1.0 - ab[1:])
    w2 = np.sqrt(alphas[1:]) * (1.0 - ab_prev[1:]) / (1.0 - ab[1:])
    beta_tilde = (1.0 - ab_prev[1:]) / (1.0 - ab[1:]) * betas[1:]
    return w1, w2, beta_tilde


def predict_x0(xt, eps_pred, t, alphas_cumprod):
    return (xt - np.sqrt(1.0 - alphas_cumprod[t]) * eps_pred) / np.sqrt(alphas_cumprod[t])


def posterior_mean(x0, xt, t, w1, w2):
    # w1, w2 数组从 t=1 开始，索引 t-1
    return w1[t - 1] * x0 + w2[t - 1] * xt


def make_demo_image(size=128):
    """合成 RGB 图像，像素范围 [-1, 1]，模拟 Normalize([0.5],[0.5]) 后的 x0"""
    y, x = np.mgrid[0:size, 0:size]
    r = 2.0 * (x / size) - 1.0
    g = 2.0 * (y / size) - 1.0
    b = np.sin(x / 12.0) * np.cos(y / 12.0)
    img = np.stack([r, g, b], axis=0)  # (3, H, W)
    return np.clip(img, -1.0, 1.0).astype(np.float32)


def to_uint8(chw):
    hwc = np.transpose(chw, (1, 2, 0))
    return ((hwc + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 图 1: Beta Schedule（对应 hf 第 98 行 DDPMScheduler）
# ---------------------------------------------------------------------------
def plot_scheduler(sched):
    betas = sched["betas"]
    ab = sched["alphas_cumprod"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.plot(betas, color="tomato", lw=1.5)
    ax1.set_xlabel("timestep t (0-indexed, HF style)")
    ax1.set_ylabel(r"$\beta_t$")
    ax1.set_title(r"Beta Schedule: linspace($10^{-4}$, 0.02, 1000)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ab, label=r"$\bar\alpha_t$", color="steelblue")
    ax2.plot(np.sqrt(ab), "--", label=r"$\sqrt{\bar\alpha_t}$ (signal)", color="orange")
    ax2.plot(np.sqrt(1 - ab), ":", label=r"$\sqrt{1-\bar\alpha_t}$ (noise)", color="green")
    ax2.set_xlabel("timestep t")
    ax2.set_ylabel("value")
    ax2.set_title("Signal vs Noise weights (used in add_noise)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / "01_scheduler.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 图 2: 闭合公式 vs 逐步迭代（对应 add_noise 的数学基础）
# ---------------------------------------------------------------------------
def plot_closed_form_verify(sched):
    alphas = sched["alphas"]
    betas = sched["betas"]
    ab = sched["alphas_cumprod"]
    target_t = 500  # HF 0-index: timestep=499 等价于第 500 步

    x0 = np.array([3.0])
    np.random.seed(0)

    x = x0.copy()
    for i in range(target_t):
        x = forward_step(x, i, alphas, betas)
    x_step = float(x[0])

    n_samples = 10000
    eps_batch = np.random.randn(n_samples)
    t_idx = target_t - 1
    x_closed = np.sqrt(ab[t_idx]) * x0[0] + np.sqrt(1 - ab[t_idx]) * eps_batch

    mu = np.sqrt(ab[t_idx]) * x0[0]
    sigma = np.sqrt(1 - ab[t_idx])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(x_closed, bins=60, density=True, alpha=0.75, color="steelblue", label="closed-form samples")
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(xs, norm.pdf(xs, mu, sigma), "r-", lw=2, label=rf"Theory $\mathcal{{N}}({mu:.2f}, {sigma:.2f}^2)$")
    ax.axvline(x_step, color="orange", lw=2, ls="--", label=f"step-by-step = {x_step:.3f}")
    ax.set_title(f"Closed-form vs step-by-step at t={target_t} (HF index {t_idx})")
    ax.set_xlabel(r"$x_t$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "02_closed_form_verify.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 图 3: 后验均值权重 w1, w2 与 beta_tilde（对应 DDPMScheduler.step 内部）
# ---------------------------------------------------------------------------
def plot_posterior(sched):
    w1, w2, beta_tilde = posterior_params(sched)
    betas = sched["betas"]
    t_range = np.arange(1, T)

    fig = plt.figure(figsize=(12, 4.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_range, w1, label=r"$w_1(t)$ weight on $x_0$", color="green")
    ax1.plot(t_range, w2, label=r"$w_2(t)$ weight on $x_t$", color="purple")
    ax1.set_xlabel("timestep t (1-indexed)")
    ax1.set_ylabel("weight")
    ax1.set_title(r"Posterior mean $\tilde\mu_t = w_1 x_0 + w_2 x_t$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_range, beta_tilde, color="darkorange", label=r"$\tilde\beta_t$")
    ax2.plot(t_range, betas[1:], color="gray", ls="--", label=r"$\beta_t$")
    ax2.set_xlabel("timestep t (1-indexed)")
    ax2.set_ylabel("variance")
    ax2.set_title(r"Posterior variance $\tilde\beta_t \leq \beta_t$")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    path = OUT_DIR / "03_posterior.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 图 4: 合成图像正向加噪（对应 hf 第 98-106 行）
# ---------------------------------------------------------------------------
def plot_image_forward_noise(sched):
    x0 = make_demo_image(128)[None, ...]  # (1, 3, H, W)
    ab = sched["alphas_cumprod"]
    timesteps_show = [0, 50, 200, 500, 800, 999]
    noise = np.random.randn(*x0.shape)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for i, t in enumerate(timesteps_show):
        ax = axes[i]
        if t == 0:
            xt = x0[0]
            sqrt_ab = 1.0
        else:
            xt = add_noise(x0, noise, t, ab)[0]
            sqrt_ab = np.sqrt(ab[t])

        ax.imshow(to_uint8(xt))
        ax.set_title(
            f"timestep={t}\n"
            rf"$\sqrt{{\bar\alpha_t}}$={sqrt_ab:.3f}, "
            rf"noise ratio=$1-\bar\alpha_t$={1-ab[t if t>0 else 0]:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(
        "Forward diffusion on synthetic x0 (same formula as noise_scheduler.add_noise)",
        fontsize=12,
        y=1.02,
    )
    path = OUT_DIR / "04_image_forward_noise.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 图 5: add_noise 手写 vs 逐步迭代 像素级误差
# ---------------------------------------------------------------------------
def plot_add_noise_equivalence(sched):
    x0 = make_demo_image(64)[None, ...]
    noise = np.random.randn(*x0.shape)
    alphas = sched["alphas"]
    betas = sched["betas"]
    ab = sched["alphas_cumprod"]

    ts = [10, 100, 500, 999]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))

    for col, t in enumerate(ts):
        closed = add_noise(x0, noise, t, ab)[0]

        x = x0.copy()
        for i in range(t):
            x = forward_step(x, i, alphas, betas)
        stepped = x[0]

        diff = np.abs(closed - stepped).mean()

        axes[0, col].imshow(to_uint8(closed))
        axes[0, col].set_title(f"t={t}\nclosed-form", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(to_uint8(stepped))
        axes[1, col].set_title(f"step-by-step\nmean |diff|={diff:.2e}", fontsize=9)
        axes[1, col].axis("off")

    fig.suptitle("add_noise (closed) == iterative forward (same noise seed path when t steps use same eps chain...)\n"
                 "Note: for image demo we compare distribution-equivalent paths; scalar verify in fig 2.", fontsize=10)
    path = OUT_DIR / "05_add_noise_equivalence.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 图 6: 训练一步数据流（对应 train_loop 第 184-202 行）
# ---------------------------------------------------------------------------
def plot_training_flow():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.3, 1.8, "clean_images\n= $x_0$", "#E8F4FD"),
        (2.0, 2.6, "noise $\\sim\\mathcal{N}(0,I)$", "#FFF3E0"),
        (2.0, 1.0, "timesteps $\\sim U\\{0..999\\}$", "#FFF3E0"),
        (3.8, 1.8, "add_noise\n$x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\varepsilon$", "#E8F5E9"),
        (6.0, 1.8, "UNet2DModel\n$\\varepsilon_\\theta(x_t,t)$", "#F3E5F5"),
        (8.0, 1.8, "MSE($\\varepsilon$, $\\varepsilon_\\theta$)", "#FFEBEE"),
    ]
    for x, y, text, color in boxes:
        ax.text(
            x, y, text, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="#555"),
        )

    arrows = [
        ((1.1, 1.8), (2.5, 2.6)),
        ((1.1, 1.8), (2.5, 1.0)),
        ((2.8, 2.4), (3.4, 2.0)),
        ((2.8, 1.2), (3.4, 1.6)),
        ((5.2, 1.8), (5.6, 1.8)),
        ((7.0, 1.8), (7.5, 1.8)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_title("One training step in hf_diffusion_train.py (train_loop L184-202)", fontsize=13, pad=12)
    path = OUT_DIR / "06_training_flow.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 图 7: 反向单步（对应 DDPMPipeline / scheduler.step）
# ---------------------------------------------------------------------------
def plot_reverse_step(sched):
    w1, w2, _ = posterior_params(sched)
    ab = sched["alphas_cumprod"]

    x0_true = make_demo_image(64)
    t = 500
    noise = np.random.randn(1, *x0_true.shape)
    xt = add_noise(x0_true[None], noise, t, ab)[0]

    # 完美预测噪声 -> 完美 x0_hat
    eps_perfect = noise[0]
    x0_hat = predict_x0(xt, eps_perfect, t, ab)
    mu_tilde = posterior_mean(x0_true, xt, t, w1, w2)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    titles = [
        f"$x_t$ (t={t})",
        r"$\hat x_0$ from perfect $\varepsilon$",
        r"$\tilde\mu_t = w_1 x_0 + w_2 x_t$",
        r"$x_0$ ground truth",
    ]
    imgs = [xt, x0_hat, mu_tilde, x0_true]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(to_uint8(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        "Reverse step math (sampling): predict noise -> recover x0_hat -> posterior mean -> x_{t-1}",
        fontsize=11,
        y=1.05,
    )
    path = OUT_DIR / "07_reverse_step.png"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def print_numeric_table(sched):
    ab = sched["alphas_cumprod"]
    w1, w2, beta_tilde = posterior_params(sched)
    rows = [0, 50, 200, 500, 800, 999]
    print("\n" + "=" * 72)
    print("HF 0-index timestep | sqrt(ab) | sqrt(1-ab) | w1 (approx) | w2 (approx)")
    print("-" * 72)
    for t in rows:
        w1v = w1[t - 1] if t > 0 else float("nan")
        w2v = w2[t - 1] if t > 0 else float("nan")
        print(
            f"  t={t:4d}            | {np.sqrt(ab[t]):.4f}   | "
            f"{np.sqrt(1-ab[t]):.4f}     | {w1v:.4f}      | {w2v:.4f}"
        )
    print("=" * 72)


def main():
    sched = make_schedule()
    plot_scheduler(sched)
    plot_closed_form_verify(sched)
    plot_posterior(sched)
    plot_image_forward_noise(sched)
    plot_add_noise_equivalence(sched)
    plot_training_flow()
    plot_reverse_step(sched)
    print_numeric_table(sched)
    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
