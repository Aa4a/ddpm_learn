# Section 08 - 离散 Mask 扩散（通往 LLaDA）

> 衔接：连续 DDPM（§01–05）+ 双向 Transformer（§06 Encoder）→ 文本上的 Masked Diffusion  
> 目标论文：Nie et al., *Large Language Diffusion Models*（LLaDA）, 2025

## 本节目标

- 理解 **为什么** 文本不能直接套用高斯加噪
- 掌握 **正向随机 Mask**、**双向预测**、**迭代 Unmask** 三条主线
- 在短句上跑通迷你 Masked Diffusion LM（教学版，非 LLaDA 8B）
- 能把本节概念 **对号入座** 到 LLaDA 论文

## 本节目录

- 零、8 节学习路线总览
- 一～八、各小节说明
- 九、文件与环境
- 十、与前序章节 / LLaDA 的联系

---

## 零、8 节学习路线总览

```
08.1  连续 vs 离散：为什么用 <mask>
  ↓
08.2  正向：按比例 t 随机抹词
  ↓
08.3  必须双向：不能 Causal
  ↓
08.4  损失：mask 位 CE（+ 1/t）
  ↓
08.5  组装 MaskPredictor，验维度
  ↓
08.6  短句上训练
  ↓
08.7  迭代 Unmask 采样
  ↓
08.8  对照 LLaDA，下一步读什么
```

| 小节 | 脚本 | 核心问题 | 预计耗时 |
|------|------|----------|----------|
| 08.1 | `01_discrete_vs_continuous.py` | 文本噪声是什么？ | 1 分钟 |
| 08.2 | `02_forward_masking.py` | t 如何控制破坏程度？ | 1 分钟 |
| 08.3 | `03_bidirectional_vs_causal.py` | 为何不用因果掩码？ | 1 分钟 |
| 08.4 | `04_training_objective.py` | 损失怎么写？ | 1 分钟 |
| 08.5 | `05_model_assembly.py` | shape 对不对？ | 1 分钟 |
| 08.6 | `06_train.py` | 能否训出可填空的模型？ | CPU `--fast` 约 1 分钟 |
| 08.7 | `07_sample.py` | 如何从全 mask 生成？ | 1 分钟 |
| 08.8 | `08_llada_bridge.py` | 和 LLaDA / BERT / AR 啥关系？ | 1 分钟 |

```bash
conda activate ddpm_learn
cd section08_masked_diffusion

python 01_discrete_vs_continuous.py
python 02_forward_masking.py
python 03_bidirectional_vs_causal.py
python 04_training_objective.py
python 05_model_assembly.py
python 06_train.py --fast
python 07_sample.py
python 07_sample.py --prompt "i love"
python 08_llada_bridge.py
```

---

## 一、08.1 连续 vs 离散

图像像素可连续插值；词表 id 不能。  
Masked Diffusion 把「最大噪声」定义为特殊符号 **`<mask>`**（吸收态）。

| | 连续 DDPM | Mask 扩散 |
|--|-----------|-----------|
| 噪声 | 高斯 ε | `<mask>` 替换 |
| 预测 | ε / x0 | 原 token |
| 损失 | MSE | CrossEntropy |

---

## 二、08.2 正向 Mask

每个可破坏位置独立：以概率 $t$ 变成 `<mask>`。  
$t=0$ 完好，$t=1$ 内容全遮（本教学版默认保护 `<bos>/<eos>`）。

运行后看 `figures/02_forward_masking.png`。

---

## 三、08.3 双向 vs 因果

猜 `i <mask> you` 需要左右文 → **Encoder-only、无 Causal Mask**。  
与 §06 自回归 Decoder 对照着理解。

---

## 四、08.4 训练目标

```
L ≈ E_{t ~ U(ε,1)} [ (1/t) · CE(x₀^mask, p_θ(· | x_t)) ]
```

只在被抹掉的位置算 CE；`1/t` 加权贴近常见 Masked Diffusion / LLaDA 写法。

---

## 五、08.5 组装模型

`MaskPredictor`：Embedding + 正弦 PE + Transformer Encoder × N + Linear → logits $(B,L,V)$。

---

## 六、08.6 训练

```bash
python 06_train.py --fast
```

输出：`checkpoints/last.pth`、`figures/06_loss_curve.png`。

内置约 80 句英文（来自 §06 语料英文侧），目的是 **过拟合短句、跑通流程**。

---

## 七、08.7 采样

```bash
python 07_sample.py
python 07_sample.py --prompt "i love" --seq-len 6 --steps 4
```

流程：全 mask → 多步预测 → **高置信揭开、低置信 remask** → 成句。  
`--prompt` 演示「条件位始终可见」（LLaDA SFT 的迷你版直觉）。

---

## 八、08.8 通向 LLaDA

读完本节再看论文时，重点对齐：

- absorbing mask 过程  
- 可变掩码率预训练  
- SFT 只 mask response  
- low-confidence remasking  

论文：[arXiv:2502.09992](https://arxiv.org/abs/2502.09992) · [项目页](https://ml-gsai.github.io/LLaDA-demo/)

---

## 九、文件说明与环境

| 文件 / 目录 | 说明 |
|-------------|------|
| `01_…` ~ `08_…` | 分节脚本 |
| `md_model.py` | MaskPredictor、q_sample、loss、sample |
| `data_utils.py` | 短句语料、词表（含 `<mask>`） |
| `checkpoints/` | 权重（建议 gitignore） |
| `figures/` | 可视化 |

```bash
conda activate ddpm_learn
cd section08_masked_diffusion
```

依赖与 §06 相同：`torch`, `matplotlib`, `tqdm`。

---

## 十、与前序章节的联系

| 概念 | §01–05 DDPM | §06 Transformer | §08 Mask 扩散 |
|------|-------------|-----------------|---------------|
| 「噪声」 | 高斯 | — | `<mask>` |
| 网络 | U-Net | Enc-Dec（Causal Dec） | Encoder-only |
| 损失 | MSE(ε) | CE（下一词） | CE（mask 位） |
| 采样 | 逐步去噪 | 自回归 | 并行 Unmask |
| 生成顺序 | 空间并行 | 左→右 | 多位置并行 |

§07（VAE / CFG）对读 LLaDA **不是前置**；学完 §06 + 本节即可进论文。
