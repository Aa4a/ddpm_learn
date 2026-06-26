# %% [markdown]
# # 一步采样：从原图直接得到任意 t 的噪声图
# 对比闭合公式 vs 逐步迭代，证明两者等价

# %% 导入
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(2024)

# %% 超参数
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_cumprod = np.cumprod(alphas)

# %% 闭合公式：一步得到 x_t
def forward_diffusion(x0, t):
    sqrt_alpha_bar = np.sqrt(alpha_cumprod[t])
    sqrt_one_minus = np.sqrt(1 - alpha_cumprod[t])
    eps = np.random.randn(*x0.shape)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * eps
    return xt, eps

# %% 逐步迭代：从 x0 一步一步走到 x_t
def forward_step_by_step(x0, t):
    x = x0.copy()
    for i in range(t):
        eps = np.random.randn(*x.shape)
        x = np.sqrt(alphas[i]) * x + np.sqrt(betas[i]) * eps
    return x

# %% 构造一个"原始图像"：二维高斯混合点云
n_points = 800
# 三个高斯簇，像一张简单图片的像素分布
mean1 = np.array([-1.5, 1.0])
mean2 = np.array([1.5, 0.5])
mean3 = np.array([0.0, -1.5])

x0 = np.concatenate([
    np.random.randn(n_points // 3, 2) * 0.2 + mean1,
    np.random.randn(n_points // 3, 2) * 0.2 + mean2,
    np.random.randn(n_points // 3, 2) * 0.2 + mean3,
])

# %% 选择展示的时间步
timesteps = [0, 100, 300, 500, 700, 999]

# %% 绘图对比
fig = plt.figure(figsize=(14, 8))

gs = gridspec.GridSpec(2, len(timesteps), figure=fig, hspace=0.35, wspace=0.25)

for idx, t in enumerate(timesteps):
    # 上排：闭合公式（直接一步）
    ax1 = fig.add_subplot(gs[0, idx])
    if t == 0:
        xt_one = x0.copy()
    else:
        xt_one, _ = forward_diffusion(x0, t - 1)
    ax1.scatter(xt_one[:, 0], xt_one[:, 1], s=5, alpha=0.6, c='steelblue')
    ax1.set_xlim(-4, 4); ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_title(f't={t}\n闭合公式一步', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 下排：逐步迭代
    ax2 = fig.add_subplot(gs[1, idx])
    if t == 0:
        xt_iter = x0.copy()
    else:
        xt_iter = forward_step_by_step(x0, t)
    ax2.scatter(xt_iter[:, 0], xt_iter[:, 1], s=5, alpha=0.6, c='coral')
    ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_title(f't={t}\n逐步迭代', fontsize=10)
    ax2.grid(True, alpha=0.3)

fig.suptitle('一步采样 vs 逐步迭代：闭合公式等价于 t 步加噪', fontsize=14, y=1.02)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/04_one_step_sampling.png',
            dpi=120, bbox_inches='tight')
plt.show()

# %% 数值验证：计算 t=500 时两种方法的均值/方差差异
t_test = 500
xt_one_test, _ = forward_diffusion(x0, t_test - 1)
xt_iter_test = forward_step_by_step(x0, t_test)

mean_diff = np.abs(xt_one_test.mean(axis=0) - xt_iter_test.mean(axis=0))
std_diff = np.abs(xt_one_test.std(axis=0) - xt_iter_test.std(axis=0))

print("=" * 55)
print(f"t={t_test} 时两种方法对比：")
print(f"  均值差异: {mean_diff}")
print(f"  标准差差异: {std_diff}")
print(f"  理论均值: {np.sqrt(alpha_cumprod[t_test-1]) * x0.mean(axis=0)}")
print(f"  理论标准差: {np.sqrt(1 - alpha_cumprod[t_test-1])}")
print("=" * 55)

# %% 绘制一个信号权重变化的曲线图
fig2, ax = plt.subplots(figsize=(8, 4))
t_range = np.arange(1, T + 1)
ax.plot(t_range, np.sqrt(alpha_cumprod), label=r'$\sqrt{\bar\alpha_t}$（原图信号权重）',
        color='royalblue', lw=2)
ax.plot(t_range, np.sqrt(1 - alpha_cumprod), label=r'$\sqrt{1-\bar\alpha_t}$（噪声权重）',
        color='tomato', lw=2)

for t in timesteps[1:]:
    ax.axvline(t, color='gray', linestyle=':', alpha=0.4)
    ax.text(t, 0.05, f't={t}', rotation=90, fontsize=8, ha='center')

ax.set_xlabel('timestep t')
ax.set_ylabel('权重')
ax.set_title('闭合公式中信号与噪声的权重随时间变化')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, T)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/04_weight_curve.png',
            dpi=120, bbox_inches='tight')
plt.show()
print("图片已保存：04_one_step_sampling.png, 04_weight_curve.png")
