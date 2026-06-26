# %% [markdown]
# # Section 01 - 数学原理可视化
# 验证闭合公式 & 展示后验均值 tilde_mu

# %% 导入与超参数
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from scipy.stats import norm

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(0)

T = 1000
betas         = np.linspace(1e-4, 0.02, T)
alphas        = 1.0 - betas
alpha_cumprod = np.cumprod(alphas)          # ᾱt

# %% 实验1：验证闭合公式与逐步迭代等价
x0 = np.array([3.0])          # 一维标量，便于验证
target_t = 500

# 方法A：逐步迭代 T=500 步
x = x0.copy()
for i in range(target_t):
    eps = np.random.randn(*x.shape)
    x = np.sqrt(alphas[i]) * x + np.sqrt(betas[i]) * eps
x_step = float(x)

# 方法B：闭合公式，一步采样 10000 次，验证分布一致
n_samples = 10000
eps_batch = np.random.randn(n_samples)
x_closed  = np.sqrt(alpha_cumprod[target_t-1]) * x0[0] + np.sqrt(1 - alpha_cumprod[target_t-1]) * eps_batch

theoretical_mean = np.sqrt(alpha_cumprod[target_t-1]) * x0[0]
theoretical_std  = np.sqrt(1 - alpha_cumprod[target_t-1])

# %% 实验2：后验均值权重 w1, w2 与后验方差
alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])  # ᾱ_{t-1}
w1 = np.sqrt(alpha_cumprod_prev[1:]) * betas[1:] / (1 - alpha_cumprod[1:])
w2 = np.sqrt(alphas[1:]) * (1 - alpha_cumprod_prev[1:]) / (1 - alpha_cumprod[1:])
beta_tilde = (1 - alpha_cumprod_prev[1:]) / (1 - alpha_cumprod[1:]) * betas[1:]

# %% 绘图：四子图汇总
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- 图1：闭合公式采样分布 vs 理论值 ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(x_closed, bins=60, density=True, alpha=0.7, color='steelblue', label='Closed-form samples')
x_plot = np.linspace(theoretical_mean - 4*theoretical_std,
                     theoretical_mean + 4*theoretical_std, 300)
ax1.plot(x_plot, norm.pdf(x_plot, theoretical_mean, theoretical_std),
         'r-', lw=2, label=f'Theory N({theoretical_mean:.2f}, {theoretical_std:.2f}^2)')
ax1.axvline(x_step, color='orange', lw=2, linestyle='--', label=f'Step-by-step={x_step:.2f}')
ax1.set_title(f'Closed-form vs Step-by-step at t={target_t}')
ax1.set_xlabel('x_t value'); ax1.set_ylabel('Density')
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# --- 图2：sqrt(alpha_bar) & sqrt(1-alpha_bar) 随 t 变化 ---
ax2 = fig.add_subplot(gs[0, 1])
t_range = np.arange(1, T+1)
ax2.plot(t_range, np.sqrt(alpha_cumprod), label=r'$\sqrt{\bar\alpha_t}$ (signal weight)', color='royalblue')
ax2.plot(t_range, np.sqrt(1 - alpha_cumprod), label=r'$\sqrt{1-\bar\alpha_t}$ (noise weight)', color='tomato')
ax2.axhline(0.5, color='gray', linestyle=':', lw=1)
ax2.set_xlabel('timestep t'); ax2.set_ylabel('coefficient')
ax2.set_title('Signal vs Noise Weights over Time')
ax2.legend(); ax2.grid(True, alpha=0.3)

# --- 图3：后验均值的两项权重 ---
ax3 = fig.add_subplot(gs[1, 0])
t_range2 = np.arange(2, T+1)
ax3.plot(t_range2, w1, label=r'$w_1(t)$: weight on $x_0$', color='green')
ax3.plot(t_range2, w2, label=r'$w_2(t)$: weight on $x_t$', color='purple')
ax3.set_xlabel('timestep t'); ax3.set_ylabel('weight')
ax3.set_title(r'Posterior Mean $\tilde\mu_t = w_1 x_0 + w_2 x_t$')
ax3.legend(); ax3.grid(True, alpha=0.3)

# --- 图4：posterior variance tilde_beta ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_range2, beta_tilde, color='darkorange')
ax4.plot(t_range2, betas[1:], color='gray', linestyle='--', label=r'$\beta_t$')
ax4.set_xlabel('timestep t'); ax4.set_ylabel('variance')
ax4.set_title(r'Posterior Variance $\tilde\beta_t \leq \beta_t$')
ax4.legend(); ax4.grid(True, alpha=0.3)

plt.suptitle('DDPM Math Verification', fontsize=14, y=1.01)
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/02_math_visualization.png',
            dpi=120, bbox_inches='tight')
plt.show()

# %% 打印数值验证
print("=" * 55)
print(f"  Closed-form mean  : {x_closed.mean():.4f}  (theory: {theoretical_mean:.4f})")
print(f"  Closed-form std   : {x_closed.std():.4f}  (theory: {theoretical_std:.4f})")
print()
print(f"  At t=2:  w1={w1[0]:.4f}, w2={w2[0]:.4f}, sum={w1[0]+w2[0]:.4f}")
print(f"  At t=500: w1={w1[498]:.4f}, w2={w2[498]:.4f}")
print(f"  At t=999: w1={w1[-1]:.4f}, w2={w2[-1]:.4f}")
print()
print("  -> Early steps: x0 matters more (high w1)")
print("  -> Late  steps: xt dominates    (high w2)")
print("=" * 55)
