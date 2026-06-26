# %% [markdown]
# # 独立高斯相加：方差相加原理可视化
# 验证 a*eps1 + b*eps2 ~ N(0, a^2+b^2)

# %% 导入
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import norm

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# %% 参数设置
a = 0.8
b = 0.6
n = 100000

# 独立标准高斯
eps1 = np.random.randn(n)
eps2 = np.random.randn(n)

# 线性组合
z = a * eps1 + b * eps2

# 理论分布
theoretical_std = np.sqrt(a**2 + b**2)
theoretical_var = a**2 + b**2
x_plot = np.linspace(-5, 5, 500)

# %% 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 图1：eps1 的分布
ax = axes[0, 0]
ax.hist(eps1, bins=80, density=True, alpha=0.7, color='skyblue', label='empirical')
ax.plot(x_plot, norm.pdf(x_plot, 0, 1), 'r-', lw=2, label='theory N(0,1)')
ax.set_title(r'$\varepsilon_1 \sim N(0,1)$')
ax.set_xlabel('value'); ax.set_ylabel('density')
ax.legend(); ax.grid(True, alpha=0.3)

# 图2：eps2 的分布
ax = axes[0, 1]
ax.hist(eps2, bins=80, density=True, alpha=0.7, color='lightgreen', label='empirical')
ax.plot(x_plot, norm.pdf(x_plot, 0, 1), 'r-', lw=2, label='theory N(0,1)')
ax.set_title(r'$\varepsilon_2 \sim N(0,1)$')
ax.set_xlabel('value'); ax.set_ylabel('density')
ax.legend(); ax.grid(True, alpha=0.3)

# 图3：a*eps1 + b*eps2 的分布
ax = axes[1, 0]
ax.hist(z, bins=80, density=True, alpha=0.7, color='salmon', label='empirical')
ax.plot(x_plot, norm.pdf(x_plot, 0, theoretical_std), 'r-', lw=2,
        label=f'theory N(0, {theoretical_std:.2f}^2)')
ax.axvline(z.mean(), color='orange', linestyle='--', lw=2, label=f'mean={z.mean():.3f}')
ax.set_title(r'$Z = 0.8\varepsilon_1 + 0.6\varepsilon_2$')
ax.set_xlabel('value'); ax.set_ylabel('density')
ax.legend(); ax.grid(True, alpha=0.3)

# 图4：二维散点展示独立性
ax = axes[1, 1]
ax.scatter(eps1[::200], eps2[::200], s=3, alpha=0.4, c='steelblue')
ax.set_xlabel(r'$\varepsilon_1$'); ax.set_ylabel(r'$\varepsilon_2$')
ax.set_title('eps1 与 eps2 独立（无相关性）')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

plt.suptitle(r'独立高斯相加：$a\varepsilon_1+b\varepsilon_2 \sim N(0,a^2+b^2)$',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/03_gaussian_addition.png',
            dpi=120, bbox_inches='tight')
plt.show()

# %% 数值验证
print("=" * 55)
print(f"a = {a}, b = {b}")
print(f"理论方差: a^2 + b^2 = {a**2:.2f} + {b**2:.2f} = {theoretical_var:.4f}")
print(f"理论标准差: sqrt({theoretical_var:.4f}) = {theoretical_std:.4f}")
print(f"采样方差: {z.var():.4f}")
print(f"采样标准差: {z.std():.4f}")
print(f"采样均值: {z.mean():.4f} (理论 0)")
print(f"eps1 与 eps2 的相关系数: {np.corrcoef(eps1, eps2)[0,1]:.4f} (接近 0 = 独立)")
print("=" * 55)

# %% 验证 DDPM 中的两阶展开特例
# x_2 = sqrt(alpha1*alpha2)*x0 + sqrt(alpha2*(1-alpha1))*eps1 + sqrt(1-alpha2)*eps2
alpha1, alpha2 = 0.9999, 0.9997
a = np.sqrt(alpha2 * (1 - alpha1))
b = np.sqrt(1 - alpha2)
combined = a * eps1 + b * eps2
expected_var = a**2 + b**2
actual_var = combined.var()
print(f"\nDDPM 两阶展开验证:")
print(f"a = sqrt(alpha2*(1-alpha1)) = {a:.6f}")
print(f"b = sqrt(1-alpha2) = {b:.6f}")
print(f"理论合并方差: a^2+b^2 = {expected_var:.8f}")
print(f"采样合并方差: {actual_var:.8f}")
print(f"等价于 1 - alpha1*alpha2 = {1 - alpha1*alpha2:.8f}")
