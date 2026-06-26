# %% [markdown]
# # 一步直接到 t 步：从原图生成任意时刻的噪声图
# 用闭合公式直接展示 x0 -> x_t，t=100,300,500,700,900

# %% 导入
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK SC'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# %% 超参数
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_cumprod = np.cumprod(alphas)

# %% 闭合公式：一步得到 x_t
def forward_diffusion(x0, t):
    sqrt_alpha_bar = np.sqrt(alpha_cumprod[t])
    print("sqrt_alpha_bar.shape: ", sqrt_alpha_bar.shape)
    sqrt_one_minus = np.sqrt(1 - alpha_cumprod[t])
    eps = np.random.randn(*x0.shape)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus * eps
    return xt

# %% 构造原图：两个交叠的圆环（像 MNIST 0 和 1 的抽象）
n = 1200
theta = np.linspace(0, 2 * np.pi, n)
# 外圆
r1 = 2.0
x1 = np.stack([r1 * np.cos(theta), r1 * np.sin(theta)], axis=1)
# 内椭圆
x2 = np.stack([1.0 * np.cos(theta) + 0.5, 0.6 * np.sin(theta) + 0.3], axis=1)

x0 = np.concatenate([x1, x2]) + np.random.randn(2 * n, 2) * 0.05

# %% 选择 t 值
show_ts = [0, 100, 300, 500, 700, 999]

# %% 可视化
fig, axes = plt.subplots(1, len(show_ts), figsize=(16, 3))

for ax, t in zip(axes, show_ts):
    if t == 0:
        xt = x0.copy()
    else:
        print("x0.shape:", x0.shape)
        print("t:", t)
        xt = forward_diffusion(x0, t)
        print("xt.shape: ", xt.shape)
        print(xt)

    ax.scatter(xt[:, 0], xt[:, 1], s=4, alpha=0.5, c='steelblue')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(f't={t}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    if t == 0:
        ax.set_ylabel('y')
    else:
        ax.set_ylabel('')

# 在图上方添加箭头说明
fig.suptitle('闭合公式一步采样：x0 直接到任意 t 的噪声图', fontsize=14, y=1.05)

plt.tight_layout()
plt.savefig('/root/autodl-tmp/ddpm_learn/section01_intro/05_direct_xt.png',
            dpi=120, bbox_inches='tight')
plt.show()

print("图片已保存：05_direct_xt.png")
