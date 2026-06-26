# %% [markdown]
# # ELBO 推导与 KL 权重分析
# 从变分下界 -log p_θ(x_0) ≤ L 到各时间步的损失权重 λ_t，
# 理解 Ho et al. 为何用简化均匀权重替代理论加权。

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
alpha_cumprod = np.cumprod(alphas)                                     # ᾱ_t
alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]])       # ᾱ_{t-1}
tilde_beta = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * betas   # β̃_t

# %% ELBO 分解说明（文字输出）
print("=" * 62)
print("ELBO 完整分解（Ho et al. 2020, Eq.5）")
print("-" * 62)
print("log p_θ(x_0) >= -L，其中：")
print()
print("  L = L_T  +  Σ_{t=2}^{T} L_{t-1}  +  L_0")
print()
print("  L_T     = KL( q(x_T|x_0) || p(x_T) )")
print("            → x_T ≈ N(0,I)，与 θ 无关，训练时忽略")
print()
print("  L_{t-1} = KL( q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t) )")
print("          = (1/2σ_t²) · ||μ̃_t - μ_θ(x_t,t)||²")
print("          = λ_t · E[||ε - ε_θ(x_t,t)||²]")
print("          其中 λ_t = β_t² / (2σ_t²α_t(1-ᾱ_t))")
print()
print("  L_0     = -E[log p_θ(x_0|x_1)]  （重建项）")
print()
print("  简化损失：L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]")
print("           等价于令所有 λ_t = 1（均匀权重）")
print("=" * 62)

# %% 计算 λ_t（两种 σ_t² 选择）
# 从 t=1 开始（t=0 时 β̃_0 = 0 会造成除零）
ts = np.arange(1, T)

# 方案 A：σ_t² = β_t（正向过程方差）
#   λ_A = β_t² / (2β_t α_t(1-ᾱ_t)) = β_t / (2α_t(1-ᾱ_t))
lambda_A = betas[ts] / (2 * alphas[ts] * (1 - alpha_cumprod[ts]))

# 方案 B：σ_t² = β̃_t（后验方差，理论上更严格）
#   λ_B = β_t² / (2β̃_t α_t(1-ᾱ_t))
#        = β_t / (2α_t(1-ᾱ_{t-1}))  (化简)
lambda_B = betas[ts]**2 / (2 * tilde_beta[ts] * alphas[ts] * (1 - alpha_cumprod[ts]))

print(f"\nλ_t 范围（σ²=β_t）：    [{lambda_A.min():.4f},  {lambda_A.max():.4f}]")
print(f"λ_t 范围（σ²=β̃_t）：   [{lambda_B.min():.4f},  {lambda_B.max():.4f}]")
print(f"简化损失等价于：λ_t = 1（均匀权重，Ho et al. 实验效果更好）")

# %% 图 1：λ_t 曲线 vs 简化基线
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(ts, lambda_A, color='steelblue', lw=1.8, label=r'$\sigma_t^2 = \beta_t$（正向方差）')
ax.plot(ts, lambda_B, color='tomato',    lw=1.8, linestyle='--',
        label=r'$\sigma_t^2 = \tilde\beta_t$（后验方差）')
ax.axhline(1.0, color='#555', ls=':', lw=1.8, label='简化损失（$\lambda_t = 1$）')
ax.set_xlabel('时间步 $t$')
ax.set_ylabel(r'$\lambda_t$')
ax.set_title(r'KL 权重 $\lambda_t = \dfrac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}$')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 图 2：各时间步相对贡献（归一化后 ×T，均匀=1）
ax2 = axes[1]
# 相对于均匀权重的倍数：λ_t / (sum λ / T)
rel_A = lambda_A / (lambda_A.sum() / len(ts))
rel_B = lambda_B / (lambda_B.sum() / len(ts))

ax2.fill_between(ts, rel_A, alpha=0.45, color='steelblue',
                 label=r'$\sigma^2=\beta_t$')
ax2.fill_between(ts, rel_B, alpha=0.45, color='tomato',
                 label=r'$\sigma^2=\tilde\beta_t$')
ax2.axhline(1.0, color='#555', ls=':', lw=1.8, label='简化损失（均匀 = 1）')
ax2.set_xlabel('时间步 $t$')
ax2.set_ylabel('相对权重（均匀 = 1）')
ax2.set_title('各时间步对总损失的相对贡献（理论加权 vs 简化均匀）')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# 标注转折点（λ=1 交叉处）
for la, color, label in [(lambda_A, 'steelblue', 'A'), (lambda_B, 'tomato', 'B')]:
    cross = ts[np.argmin(np.abs(la - 1.0))]
    axes[0].axvline(cross, color=color, ls='-.', lw=1, alpha=0.6)

fig.suptitle('ELBO → 简化损失：权重 $\\lambda_t$ 分析', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section03_loss/01_elbo_weights.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n图片已保存：01_elbo_weights.png")

# %% 直觉总结
print("\n" + "=" * 62)
print("直觉总结：")
print("  加权损失：λ_t < 1 for large t → 高噪步贡献被低估")
print("  简化损失：λ_t = 1 → 均匀对待所有时间步")
print("  效果：简化损失使模型在高噪声步也充分训练，")
print("         实验上生成质量更好（Ho et al. 2020, Sec.3.4）")
print("=" * 62)
