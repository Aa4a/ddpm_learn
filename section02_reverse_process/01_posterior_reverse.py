# %% [markdown]
# # 反向过程后验推导：q(x_{t-1} | x_t, x_0)
# 用贝叶斯公式 + 配方法解析推导后验高斯，并用蒙特卡洛验证 + 反向采样演示

# %% 导入
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# %% 超参数与 schedule
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_cumprod = np.cumprod(alphas)
# alpha_cumprod_prev[t] = ᾱ_{t-1}，约定 ᾱ_0 = 1
alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])


# %% 解析后验：q(x_{t-1} | x_t, x_0) = N(μ̃_t, β̃_t)
# 配方法结论：
#   1/β̃_t = α_t/β_t + 1/(1-ᾱ_{t-1})  ->  β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
#   μ̃_t   = [√ᾱ_{t-1} β_t /(1-ᾱ_t)] x_0 + [√α_t (1-ᾱ_{t-1})/(1-ᾱ_t)] x_t
def posterior_beta(t):
    return (1 - alpha_cumprod_prev[t]) / (1 - alpha_cumprod[t]) * betas[t]


def posterior_mean(x0, xt, t):
    coef_x0 = np.sqrt(alpha_cumprod_prev[t]) * betas[t] / (1 - alpha_cumprod[t])
    coef_xt = np.sqrt(alphas[t]) * (1 - alpha_cumprod_prev[t]) / (1 - alpha_cumprod[t])
    return coef_x0 * x0 + coef_xt * xt


# %% 实例 1：具体数值，给定 x0 与 x_t 求后验
t = 100
x0_scalar = 2.0
# 用闭合公式造一个真实的 x_t（带噪），eps 取一次具体采样
eps_true = 0.7
xt_scalar = np.sqrt(alpha_cumprod[t]) * x0_scalar + np.sqrt(1 - alpha_cumprod[t]) * eps_true

mu_tilde = posterior_mean(x0_scalar, xt_scalar, t)
beta_tilde = posterior_beta(t)

print("=" * 60)
print(f"实例：t={t}, x0={x0_scalar}, x_t={xt_scalar:.4f}")
print(f"  解析后验均值 μ̃_t   = {mu_tilde:.6f}")
print(f"  解析后验方差 β̃_t   = {beta_tilde:.8f}")
print(f"  解析后验标准差 √β̃_t = {np.sqrt(beta_tilde):.6f}")
print("=" * 60)


# %% 验证：蒙特卡洛（重要性采样）逼近后验
# 思路：从先验 q(x_{t-1}|x_0)=N(√ᾱ_{t-1} x0, 1-ᾱ_{t-1}) 采样大量 x_{t-1}，
#       再用似然 q(x_t|x_{t-1})=N(√α_t x_{t-1}, β_t) 作为权重，
#       加权统计量应当收敛到解析后验 μ̃_t, β̃_t。
N = 2_000_000
prior_mean = np.sqrt(alpha_cumprod_prev[t]) * x0_scalar
prior_std = np.sqrt(1 - alpha_cumprod_prev[t])
samples = np.random.randn(N) * prior_std + prior_mean

# 似然权重（高斯核），减去最大值防溢出
log_w = -0.5 * (xt_scalar - np.sqrt(alphas[t]) * samples) ** 2 / betas[t]
w = np.exp(log_w - log_w.max())
w /= w.sum()

mc_mean = np.sum(w * samples)
mc_var = np.sum(w * (samples - mc_mean) ** 2)

print("蒙特卡洛重要性采样验证：")
print(f"  MC 后验均值 = {mc_mean:.6f}  (解析 {mu_tilde:.6f})")
print(f"  MC 后验方差 = {mc_var:.8f}  (解析 {beta_tilde:.8f})")
print("=" * 60)


# %% 实例 2：μ̃_t 的"噪声形式"等价
# 代入 x_0 = (x_t - √(1-ᾱ_t) ε)/√ᾱ_t，μ̃_t 可改写为：
#   μ̃_t = 1/√α_t * ( x_t - β_t/√(1-ᾱ_t) * ε )
mu_noise_form = (1 / np.sqrt(alphas[t])) * (
    xt_scalar - betas[t] / np.sqrt(1 - alpha_cumprod[t]) * eps_true
)
print("噪声形式等价验证：")
print(f"  x0/xt 形式 μ̃_t = {mu_tilde:.6f}")
print(f"  噪声形式  μ̃_t = {mu_noise_form:.6f}")
print("=" * 60)


# %% 图 1：后验权重 + 方差随时间变化
ts = np.arange(1, T)
coef_x0 = np.sqrt(alpha_cumprod_prev[ts]) * betas[ts] / (1 - alpha_cumprod[ts])
coef_xt = np.sqrt(alphas[ts]) * (1 - alpha_cumprod_prev[ts]) / (1 - alpha_cumprod[ts])
beta_tilde_curve = (1 - alpha_cumprod_prev[ts]) / (1 - alpha_cumprod[ts]) * betas[ts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ax1.plot(ts, coef_x0, color='royalblue', lw=2, label=r'$x_0$ 权重')
ax1.plot(ts, coef_xt, color='tomato', lw=2, label=r'$x_t$ 权重')
ax1.set_xlabel('timestep t'); ax1.set_ylabel('后验均值权重')
ax1.set_title(r'$\tilde\mu_t$ 中 $x_0$ 与 $x_t$ 的权重')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(ts, beta_tilde_curve, color='seagreen', lw=2, label=r'$\tilde\beta_t$（后验方差）')
ax2.plot(ts, betas[ts], color='gray', lw=1.5, ls='--', label=r'$\beta_t$（正向方差）')
ax2.set_xlabel('timestep t'); ax2.set_ylabel('方差')
ax2.set_title(r'后验方差 $\tilde\beta_t$ vs 正向方差 $\beta_t$')
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle('后验高斯参数随时间变化', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section02_reverse_process/01_posterior_weights.png',
            dpi=120, bbox_inches='tight')
plt.show()


# %% 图 2：单点后验分布——解析高斯 vs 蒙特卡洛直方图
fig, ax = plt.subplots(figsize=(8, 4.5))
resample_idx = np.random.choice(N, size=200_000, p=w)
post_samples = samples[resample_idx]
ax.hist(post_samples, bins=120, density=True, alpha=0.5, color='steelblue',
        label='蒙特卡洛后验样本')

grid = np.linspace(mu_tilde - 4 * np.sqrt(beta_tilde), mu_tilde + 4 * np.sqrt(beta_tilde), 400)
pdf = np.exp(-0.5 * (grid - mu_tilde) ** 2 / beta_tilde) / np.sqrt(2 * np.pi * beta_tilde)
ax.plot(grid, pdf, color='crimson', lw=2.5,
        label='解析后验 $\\mathcal{N}(\\tilde\\mu_t,\\tilde\\beta_t)$')
ax.axvline(mu_tilde, color='crimson', ls=':', alpha=0.7)
ax.axvline(x0_scalar, color='green', ls='--', alpha=0.7, label=f'真实 $x_0$={x0_scalar}')
ax.set_xlabel(r'$x_{t-1}$'); ax.set_ylabel('概率密度')
ax.set_title(f'后验 $q(x_{{t-1}}|x_t,x_0)$（t={t}）：解析 vs 蒙特卡洛')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section02_reverse_process/01_posterior_dist.png',
            dpi=120, bbox_inches='tight')
plt.show()


# %% 图 3：反向采样链演示（已知 x0 的"真后验去噪"）
# 构造 2D 原始点云作为 x_0，正向加噪到 x_T，再用真后验逐步去噪回 x_0。
# 这说明：若能拿到正确的后验均值，反向过程确实能恢复数据分布。
n_pts = 1500
ang = np.linspace(0, 2 * np.pi, n_pts)
x0_cloud = np.stack([2.0 * np.cos(ang), 2.0 * np.sin(ang)], axis=1)
x0_cloud += np.random.randn(n_pts, 2) * 0.05

# 正向：一步到 x_T
T_show = 999
eps0 = np.random.randn(*x0_cloud.shape)
xT = np.sqrt(alpha_cumprod[T_show]) * x0_cloud + np.sqrt(1 - alpha_cumprod[T_show]) * eps0

# 反向：用真后验 q(x_{t-1}|x_t, x0) 逐步采样
def reverse_step(xt, x0, t):
    mu = posterior_mean(x0, xt, t)
    if t == 0:
        return mu
    sigma = np.sqrt(posterior_beta(t))
    return mu + sigma * np.random.randn(*xt.shape)

snapshots = {}
record_ts = [999, 700, 400, 200, 50, 0]
x = xT.copy()
for step in range(T_show, -1, -1):
    if step in record_ts:
        snapshots[step] = x.copy()
    x = reverse_step(x, x0_cloud, step)

fig, axes = plt.subplots(1, len(record_ts), figsize=(16, 3))
for ax, st in zip(axes, record_ts):
    pts = snapshots[st]
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5, c='mediumvioletred')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(f't={st}', fontsize=11)
    ax.grid(True, alpha=0.3)

fig.suptitle('反向采样链：用真后验从纯噪声 x_T 逐步去噪回 x_0', fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section02_reverse_process/01_reverse_chain.png',
            dpi=120, bbox_inches='tight')
plt.show()

print("图片已保存：01_posterior_weights.png, 01_posterior_dist.png, 01_reverse_chain.png")
