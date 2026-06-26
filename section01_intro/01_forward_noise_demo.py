# %% [markdown]
# # Section 1: DDPM 正向加噪过程可视化
# 使用 `#%%` cell 格式：Ctrl+Enter 运行当前 cell，Shift+Enter 运行并跳到下一个

# %% 导入与字体设置
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# %% 超参数与 Beta Schedule
T = 1000          # 总扩散步数（原论文使用1000）
beta_start = 1e-4 # β₁
beta_end   = 0.02 # βT

# 线性 beta schedule：β₁, β₂, ..., βT 线性递增
betas  = np.linspace(beta_start, beta_end, T)          # shape (T,)
alphas = 1.0 - betas                                   # αt = 1 - βt
alpha_cumprod = np.cumprod(alphas)                     # ᾱt = ∏ αs (s=1..t)
np.set_printoptions(suppress=True, precision=8)
print("ᾱt 前5个:", alpha_cumprod[:5])
print("ᾱt 后5个:", alpha_cumprod[-5:])

# %% 定义正向过程闭合公式
# q(xₜ | x₀) = N(xₜ ; √ᾱt · x₀, (1-ᾱt)·I)
# 即：xₜ = √ᾱt · x₀ + √(1-ᾱt) · ε,  ε ~ N(0,I)

def forward_diffusion(x0, t):
    """给定 x0 和时间步 t，直接采样 xt（闭合公式，无需逐步迭代）"""
    sqrt_alpha_bar   = np.sqrt(alpha_cumprod[t])
    sqrt_one_minus   = np.sqrt(1 - alpha_cumprod[t])
    eps = np.random.randn(*x0.shape)
    xt  = sqrt_alpha_bar * x0 + sqrt_one_minus * eps
    return xt, eps

# %% 可视化正向加噪过程
# 构造一个简单的"图片"：2D 高斯分布点云
n_points = 500
x0 = np.random.randn(n_points, 2) * 0.5   # 真实数据：集中的点云

# 选几个时间步展示
timesteps_show = [0, 50, 200, 500, 800, 999]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for i, t in enumerate(timesteps_show):
    ax = axes[i]
    if t == 0:
        xt = x0.copy()
    else:
        xt, _ = forward_diffusion(x0, t - 1)   # t 从0索引

    alpha_bar_t = alpha_cumprod[t] if t > 0 else 1.0
    noise_ratio = 1 - alpha_bar_t

    ax.scatter(xt[:, 0], xt[:, 1], s=8, alpha=0.6, c='steelblue')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_title(f't = {t}\n√ᾱₜ={np.sqrt(alpha_cumprod[t-1] if t>0 else 1):.3f}  '
                 f'noise_ratio={noise_ratio:.3f}', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('DDPM 正向加噪过程：真实数据 → 高斯噪声', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/01_forward_noise.png', dpi=120, bbox_inches='tight')
plt.show()
print("图片已保存到 01_forward_noise.png")

# %% 可视化 Beta Schedule 与信号/噪声权重
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

ax1.plot(betas, color='tomato')
ax1.set_xlabel('timestep t'); ax1.set_ylabel('βt')
ax1.set_title('Beta Schedule（线性）')
ax1.grid(True, alpha=0.3)

ax2.plot(alpha_cumprod, color='steelblue', label='ᾱt = ∏αs')
ax2.plot(np.sqrt(alpha_cumprod), color='orange', linestyle='--', label='√ᾱt（信号权重）')
ax2.plot(np.sqrt(1 - alpha_cumprod), color='green', linestyle=':', label='√(1-ᾱt)（噪声权重）')
ax2.set_xlabel('timestep t'); ax2.set_ylabel('value')
ax2.set_title('信号 vs 噪声 权重随时间的变化')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/01_schedules.png', dpi=120, bbox_inches='tight')
plt.show()
print("图片已保存到 01_schedules.png")
