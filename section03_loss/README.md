# Section 03 - 损失函数：ELBO → 简化 MSE

## 本节目标

- 从变分推断角度推导 DDPM 的训练目标（ELBO）
- 将 KL 散度项化简为均值的 MSE，再进一步化为**噪声 MSE**
- 理解 $\lambda_t$ 权重，掌握 Ho et al. 简化损失的直觉
- 数值验证 $\tilde\mu_t$ 两种形式等价，理解完整训练流程

## 本节目录

- 零、先搞清楚 $p_\theta$ 和 $q$ 是什么
- 一、ELBO 从哪里来
  - 1.1 训练目标：$\log p_\theta(x_0)$
  - 1.2 引入 $q$：把积分改写成期望
  - 1.3 为什么用 Jensen 不等式
  - 1.4 展开 ELBO：从连乘积到 KL 散度
- 二、$L_{t-1}$：KL → 均值 MSE
- 三、代入噪声形式
- 四、$\sigma_t^2$ 的两种选取
- 五、简化损失
- 六、数值验证

---

## 零、先搞清楚 $p_\theta$ 和 $q$ 是什么

本节推导会同时出现两个分布。**在谈 ELBO 之前，必须先弄清它们各自扮演什么角色**——否则 $p_\theta(x_{0:T})$、$\mu_\theta$ 等符号会像凭空冒出来一样。

### 0.1 整体图景

```
正向 q（固定，不学习）：  x0 → x1 → x2 → ... → xT ≈ N(0,I)
反向 pθ（待学习）：       xT → ... → x1 → x0
```

| 符号 | 是什么 | 学不学习 | 在哪定义的 |
|------|--------|---------|-----------|
| $q(x_{1:T}\|x_0)$ | 正向加噪过程 | ✗ 固定 | Section 01 |
| $p_\theta(x_{0:T})$ | 反向生成过程（模型） | ✓ 学 $\theta$ | 本节 0.2 |
| $q(x_{t-1}\|x_t, x_0)$ | 正向链的「真后验」 | ✗ 固定，可解析 | Section 02 |

**训练的核心矛盾**：真实反向 $q(x_{t-1}|x_t)$ 算不出来（Section 02），但加了 $x_0$ 后的 $q(x_{t-1}|x_t,x_0)$ 有闭式。我们让模型 $p_\theta(x_{t-1}|x_t)$ 去逼近这个「真后验」——这就是后面 $L_{t-1}$ KL 项的含义。

### 0.2 $p_\theta$ 是什么：反向马尔可夫链

$p_\theta$ 是**生成模型**：假设数据 $x_0$ 是从纯噪声 $x_T$ 出发、逐步去噪得到的。整条链的联合概率分解为（**反向**马尔可夫结构）：

$$p_\theta(x_{0:T}) = p(x_T)\,\prod_{t=1}^{T} p_\theta(x_{t-1}\,|\,x_t)$$

各项含义：

| 因子 | 分布 | 说明 |
|------|------|------|
| $p(x_T)$ | $\mathcal{N}(0,\,I)$ | 起点：纯高斯噪声，与 $\theta$ 无关 |
| $p_\theta(x_{t-1}\|x_t)$ | $\mathcal{N}(\mu_\theta(x_t,t),\,\sigma_t^2 I)$ | 第 $t$ 步去噪：从 $x_t$ 预测 $x_{t-1}$ |

**$\theta$ 在哪里？** 只在均值 $\mu_\theta(x_t, t)$ 里——它由神经网络输出。Ho et al. 把均值写成噪声预测形式（与 Section 02 的 $\tilde\mu_t$ 结构对齐）：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t, t)\right)$$

等价地，网络 $\varepsilon_\theta(x_t, t)$ 预测「加噪时混入的噪声 $\varepsilon$」，再反推 $x_{t-1}$ 的均值。方差 $\sigma_t^2$ 是**人为设定的常数**（通常取 $\beta_t$ 或 $\tilde\beta_t$），不学习。

**采样时**怎么用 $p_\theta$？从 $x_T \sim \mathcal{N}(0,I)$ 出发，逐步执行 $x_{t-1} \sim p_\theta(\cdot|x_t)$，最终得到 $x_0$。

### 0.3 $q$ 是什么：正向马尔可夫链（复习）

正向过程 $q$ 与 $p_\theta$ **方向相反**，且**不含** $\theta$：

$$q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t\,|\,x_{t-1}), \qquad q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\,x_{t-1},\,\beta_t I)$$

闭合公式（Section 01）：$q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$。

训练时 $q$ 有两个用途：
1. **构造训练样本**：给定真实图片 $x_0$，随机抽 $t$ 和 $\varepsilon$，一步得到 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon$
2. **变分分布**：在 ELBO 推导里作为 $q(x_{1:T}|x_0)$，把难算的积分变成期望

### 0.4 边缘分布 $p_\theta(x_0)$：我们真正要最大化的量

联合分布 $p_\theta(x_{0:T})$ 描述整条链，但训练数据只有 $x_0$。我们关心的是**边缘分布**——把隐变量 $x_1,\ldots,x_T$ 积掉：

$$p_\theta(x_0) = \int p_\theta(x_{0:T})\,dx_{1:T}$$

对数似然 $\log p_\theta(x_0)$ 就是「模型认为这张图有多合理」。**最大化它 = 让生成模型拟合真实数据**。下一节从这里出发推导 ELBO。

---

## 一、ELBO 从哪里来

### 1.1 训练目标：$\log p_\theta(x_0)$ 为什么算不出来？

上一节已定义 $p_\theta(x_{0:T})$。取对数并边缘化：

$$\log p_\theta(x_0) = \log\int p_\theta(x_{0:T})\,dx_{1:T}$$

这里的 $dx_{1:T}$ 表示对 $x_1, x_2, \ldots, x_T$ **每一个**中间变量积分——不是只积 $x_1$，而是整条链上 $T$ 个隐变量全部积掉。

为什么难算？
- 积分维度 = $T \times d$（$T=1000$ 步，每步 $d$ 维像素），极高维
- 被积函数 $p_\theta(x_{0:T})$ 里嵌着神经网络 $\varepsilon_\theta$，没有解析形式
- 无法像 $q(x_t|x_0)$ 那样写出闭合公式

→ 需要**变分推断**：引入可处理的 $q$，把目标改写成可优化下界 ELBO。

### 1.2 引入 $q$：把积分改写成期望

对 $p_\theta(x_0)$ 的积分，分子分母同乘 $q(x_{1:T}|x_0)$（Section 0.3 定义的正向链）：

$$\log p_\theta(x_0)
= \log\int q(x_{1:T}|x_0)\,\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\,dx_{1:T}
= \log\mathbb{E}_{q(x_{1:T}|x_0)}\!\left[\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

括号里的比值 $\dfrac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}$ 是**重要性权重**：按 $q$ 采一条轨迹 $(x_1,\ldots,x_T)$，看模型概率与正向概率差多少。期望 $\mathbb{E}_q[\cdot]$ 可用蒙特卡洛估计——**积分变成了可采样的期望**。

但外面还套着 $\log$。$\log$ 是非线性的：

$$\log\mathbb{E}[X] \;\neq\; \mathbb{E}[\log X]$$

期望穿不进对数，还不能直接算。

### 1.3 为什么用 Jensen 不等式？

**Jensen 不等式**（$\log$ 是凹函数）：对任意随机变量 $X$，

$$\log\mathbb{E}[X] \;\ge\; \mathbb{E}[\log X]$$

直觉：凹函数图像在弦的下方——「先取期望再取 log」≥「先取 log 再取期望」。

```
        log
         |     *  log(E[X])          ← 左边：先期望再 log（更大）
         |    /|
         |   / |
         |  /  *  E[log X]           ← 右边：先 log 再期望（更小）
         | /   |
         +-----+--------→ X
              E[X]
```

令 $X = \dfrac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}$，得到**可计算的下界**：

$$\log p_\theta(x_0)
\;\ge\;
\mathbb{E}_q\!\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]
=: \text{ELBO}$$

| 量 | 能否直接算 | 说明 |
|----|-----------|------|
| $\log p_\theta(x_0)$ | ✗ | 对 $x_{1:T}$ 高维积分 |
| $\log\mathbb{E}_q[\cdots]$ | ✗ | $\log$ 穿不进期望 |
| **ELBO** | ✓ | $\log$ 在期望**里面**，可对单条轨迹逐项展开 |

**训练策略**：最大化 ELBO（下界），真实对数似然也跟着升高。这是 VAE / DDPM 的标准做法。

> **代价**：Jensen 是不等式，ELBO 与 $\log p_\theta(x_0)$ 之间有 gap。当 $q$ 接近真实后验时 gap 更小；DDPM 的高斯设计让这个 gap 在实践中可接受。

### 1.4 展开 ELBO：从连乘积到 KL 散度

ELBO 里是 $\mathbb{E}_q\!\left[\log\dfrac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$。下面分四步把它化成 KL 散度之和。

#### 步骤 A：代入 $p_\theta$ 和 $q$ 的马尔可夫分解

由 Section 0.2 和 0.3：

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t), \qquad
q(x_{1:T}|x_0) = \prod_{t=1}^{T}q(x_t|x_{t-1})$$

取对数，比值的 log 变成求和：

$$\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}
= \log p(x_T) + \sum_{t=1}^{T}\log p_\theta(x_{t-1}|x_t) - \sum_{t=1}^{T}\log q(x_t|x_{t-1})$$

#### 步骤 B：把 $q(x_t|x_{t-1})$ 改写成后验（$t \ge 2$）

对 $t \ge 2$，用贝叶斯公式（马尔可夫性 $q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})$）：

$$q(x_t|x_{t-1}) = \frac{q(x_{t-1}|x_t,x_0)\,q(x_t|x_0)}{q(x_{t-1}|x_0)}$$

取 log 后代入 $\sum_{t=1}^{T}\log q(x_t|x_{t-1})$。注意 $t=1$ 时 $q(x_1|x_0)$ 直接保留，$t \ge 2$ 才做上述替换：

$$\sum_{t=1}^{T}\log q(x_t|x_{t-1})
= \log q(x_1|x_0) + \sum_{t=2}^{T}\Big[\log q(x_{t-1}|x_t,x_0) + \log\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}\Big]$$

#### 步骤 C：望远镜抵消

看 $\sum_{t=2}^{T}\log\dfrac{q(x_t|x_0)}{q(x_{t-1}|x_0)}$ 这一项：

$$\log\frac{q(x_2|x_0)}{q(x_1|x_0)} + \log\frac{q(x_3|x_0)}{q(x_2|x_0)} + \cdots + \log\frac{q(x_T|x_0)}{q(x_{T-1}|x_0)}
= \log q(x_T|x_0) - \log q(x_1|x_0)$$

中间项全部抵消，只剩首尾。代回步骤 B 后，$\log q(x_1|x_0)$ 也前后抵消，整理得：

$$\sum_{t=1}^{T}\log q(x_t|x_{t-1})
= \sum_{t=2}^{T}\log q(x_{t-1}|x_t,x_0) + \log q(x_T|x_0)$$

#### 步骤 D：合并同期望项为 KL 散度

把步骤 A 和步骤 C 的结果合并，再对 $q(x_{1:T}|x_0)$ 取期望。利用恒等式 $\mathbb{E}_q[\log p(a|b)] = -\mathrm{KL}(q(a|b)\|p(a|b)) + \text{const}$，同类项两两配对：

| 配对 | 变成 |
|------|------|
| $\log p(x_T)$ vs $\log q(x_T\|x_0)$ | $-D_{\rm KL}(q(x_T\|x_0)\,\|\,p(x_T)) = -L_T$ |
| $\log p_\theta(x_{t-1}\|x_t)$ vs $\log q(x_{t-1}\|x_t,x_0)$（$t=2..T$） | $-\sum_{t=2}^{T} D_{\rm KL}(q(x_{t-1}\|x_t,x_0)\,\|\,p_\theta(x_{t-1}\|x_t)) = -\sum L_{t-1}$ |
| 剩余的 $\log p_\theta(x_0\|x_1)$ | $\mathbb{E}_q[\log p_\theta(x_0\|x_1)] = -L_0$ |

最终：

$$\log p_\theta(x_0) \ge \underbrace{-D_{\rm KL}(q(x_T|x_0)\|p(x_T))}_{-L_T}
+ \sum_{t=2}^{T}\underbrace{-\,D_{\rm KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))}_{-L_{t-1}}
+ \underbrace{\mathbb{E}_q[\log p_\theta(x_0|x_1)]}_{-L_0}$$

| 项 | 含义 | 是否优化 |
|----|------|---------|
| $L_T$ | 最终噪声 $q(x_T\|x_0)$ 与先验 $p(x_T)=\mathcal{N}(0,I)$ 的 KL | ✗（与 $\theta$ 无关） |
| $L_{t-1}$ | 真后验 $q$ 与模型去噪 $p_\theta$ 的 KL | ✓（**主要训练信号**） |
| $L_0$ | 最后一步重建 $x_0$ | ✓（实践中也用 MSE） |

**直觉回顾**：$L_{t-1}$ 在问——「给定 $x_t$ 和 $x_0$，正向过程认为 $x_{t-1}$ 应该是什么分布」vs「模型认为 $x_{t-1}$ 应该是什么分布」。让两者 KL 尽量小，模型就去噪学得越好。

```python
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_cumprod = np.cumprod(alphas)                                # ᾱ_t
alpha_cumprod_prev = np.concatenate([[1.0], alpha_cumprod[:-1]]) # ᾱ_{t-1}
tilde_beta = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * betas  # β̃_t
```

---

## 二、$L_{t-1}$：KL → 均值 MSE

上一节 $L_{t-1}$ 是两个分布的 KL。现在代入它们的具体形式（Section 02 的后验 + Section 0.2 的模型）。

**$q$ 侧**（真后验，固定）：$q(x_{t-1}|x_t,x_0) = \mathcal{N}(\tilde\mu_t(x_t,x_0),\,\tilde\beta_t I)$

**$p_\theta$ 侧**（模型，可学习）：$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\,\sigma_t^2 I)$

两者方差都是**固定常数**（不学习）。**两个等方差高斯的 KL 有闭式公式**：

$$D_{\rm KL}\big(\mathcal{N}(\mu_q,\sigma^2 I)\,\|\,\mathcal{N}(\mu_p,\sigma^2 I)\big)
= \frac{1}{2\sigma^2}\|\mu_q-\mu_p\|^2$$

（同方差时，KL 公式中的 $\log\frac{\sigma_p}{\sigma_q}$ 和迹项全为零，只剩均值差项。）代入：

$$L_{t-1} = D_{\rm KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))
= \frac{1}{2\sigma_t^2}\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\|^2$$

其中 $\sigma_t^2$ 是固定常数（两种常见选取见下文）。

---

## 三、代入噪声形式

由 Section 02 的结论（$\tilde\mu_t$ 噪声形式）：

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon\right)$$

模型对应设为（Section 0.2）：

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right)$$

两个均值只有括号里的 $\varepsilon$ vs $\varepsilon_\theta$ 不同，**相减时 $x_t$ 项抵消**：

$$\tilde\mu_t - \mu_\theta
= \frac{1}{\sqrt{\alpha_t}}\!\left[-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon\right] - \frac{1}{\sqrt{\alpha_t}}\!\left[-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\right]
= \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\big(\varepsilon_\theta - \varepsilon\big)$$

取平方范数（系数平方提出，符号不影响平方）：

$$\|\tilde\mu_t - \mu_\theta\|^2 = \left(\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}\right)^2\|\varepsilon - \varepsilon_\theta\|^2 = \frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$$

因此：

$$\boxed{L_{t-1} = \underbrace{\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t(1-\bar\alpha_t)}}_{\lambda_t}\cdot\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2}$$

```python
# 方案 A：σ_t² = β_t
lambda_A = betas[ts] / (2 * alphas[ts] * (1 - alpha_cumprod[ts]))

# 方案 B：σ_t² = β̃_t（后验方差）
lambda_B = betas[ts]**2 / (2 * tilde_beta[ts] * alphas[ts] * (1 - alpha_cumprod[ts]))
```

---

## 四、$\sigma_t^2$ 的两种选取

| 选取 | 表达式 | 特点 |
|------|--------|------|
| $\sigma_t^2 = \beta_t$ | $\lambda_t = \dfrac{\beta_t}{2\alpha_t(1-\bar\alpha_t)}$ | 简单，小 $t$ 略大 |
| $\sigma_t^2 = \tilde\beta_t$ | $\lambda_t = \dfrac{\beta_t}{2\alpha_t(1-\bar\alpha_{t-1})}$ | 理论最优（后验方差） |

两种方案的 $\lambda_t$ 随 $t$ 减小（均小于 1，大噪步被低估）：

![ELBO 权重分析](01_elbo_weights.png)

> **直觉**：$\lambda_t < 1$ 意味着加权损失**低估**了大噪声步的重要性。简化损失（$\lambda_t = 1$）均匀对待所有步，实验上生成质量更好（Ho et al. 2020, Sec. 3.4）。

---

## 五、简化损失

$$\boxed{L_{\rm simple} = \mathbb{E}_{t\sim\mathcal{U}[1,T],\, x_0,\, \varepsilon\sim\mathcal{N}(0,I)}\!\left[\|\varepsilon - \varepsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon}_{x_t},\, t)\|^2\right]}$$

这就是 DDPM 的最终训练目标。训练循环极为简洁：

```python
# 一个训练步
t   = np.random.randint(0, T)                          # 随机时间步
eps = np.random.randn(*x0.shape)                       # 随机噪声
xt  = np.sqrt(alpha_cumprod[t]) * x0 + \
      np.sqrt(1 - alpha_cumprod[t]) * eps              # 加噪
loss = np.mean((eps - eps_theta(xt, t))**2)            # MSE
```

---

## 六、数值验证

### 6.1 $\tilde\mu_t$ 两种形式等价

代入 $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon)$ 代数变换严格可逆，数值误差在浮点精度内（$\sim 10^{-14}$）：

```python
def mu_x0_form(x0, xt, t):
    c0 = np.sqrt(alpha_cumprod_prev[t]) * betas[t] / (1 - alpha_cumprod[t])
    ct = np.sqrt(alphas[t]) * (1 - alpha_cumprod_prev[t]) / (1 - alpha_cumprod[t])
    return c0 * x0 + ct * xt

def mu_eps_form(xt, eps, t):
    return (1 / np.sqrt(alphas[t])) * (
        xt - betas[t] / np.sqrt(1 - alpha_cumprod[t]) * eps
    )
```

### 6.2 简化损失随模型质量的变化

令 $\varepsilon_\theta = \alpha\cdot\varepsilon_{\rm true} + (1-\alpha)\cdot\varepsilon_{\rm rand}$（$\alpha$ 表示模型质量），理论上：

$$L_{\rm simple} = (1-\alpha)^2 \cdot 2d \quad (d\text{ 为维度})$$

![简化损失验证与模型质量](02_simplified_loss.png)

### 6.3 训练批次样貌

不同时间步下，模型输入 $x_t$ 与预测目标 $\varepsilon$ 的分布：

![训练批次可视化](02_training_samples.png)

> 注意：无论 $t$ 多大，**预测目标 $\varepsilon$ 始终是 $\mathcal{N}(0,I)$**（标准正态），这使得 MSE 损失的量纲不随 $t$ 变化——简化损失的数值稳定性来源之一。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `01_elbo_weights.py` | ELBO 分解说明、$\lambda_t$ 曲线、相对贡献分析 |
| `02_noise_prediction.py` | $\tilde\mu_t$ 等价性验证、$L_{\rm simple}$ 曲线、训练批次可视化 |
| `01_elbo_weights.png` | $\lambda_t$ 曲线 + 相对贡献图 |
| `02_simplified_loss.png` | 等价性误差 + 损失曲线 |
| `02_training_samples.png` | 不同 $t$ 下 $x_t$ 与 $\varepsilon$ 的样貌 |

## 运行

```bash
conda activate ddpm_learn
python 01_elbo_weights.py
python 02_noise_prediction.py
```

---

## 下一节预告

**Section 04**：U-Net 架构 —— 时间嵌入（Sinusoidal Embedding）+ 残差 U-Net，理解 $\varepsilon_\theta(x_t, t)$ 的网络设计。
