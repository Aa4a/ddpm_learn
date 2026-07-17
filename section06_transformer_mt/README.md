# Section 06 - Transformer 机器翻译（分节教程）

> 参考论文：Vaswani et al., "Attention Is All You Need", NeurIPS 2017

## 本节目标

- 从零理解「英文 → 中文」机器翻译在做什么，**不预设 NLP 基础**
- 按 **10 个小节**循序渐进：先建立直觉，再学公式，最后训练、推理与注意力可视化
- 每一节只引入 **一个新概念**，配有可运行脚本和图示
- 跑通完整流程：内置语料 → 训练 Transformer → 交互翻译

## 本节目录

- 零、10 节学习路线总览
- 一、06.1 机器翻译在做什么？（Seq2Seq 直觉）
- 二、06.2 从文字到数字（分词与词表）
- 三、06.3 注意力机制直觉
- 四、06.4 位置编码
- 五、06.5 多头注意力与 Mask
- 六、06.6 Cross-Attention（交叉注意力）专节
- 七、06.7 组装完整 Transformer
- 八、06.8 训练（Teacher Forcing）
- 九、06.9 推理（自回归解码）
- 十、06.10 可视化 Attention（看模型「看了哪里」）
- 十一、文件说明与环境
- 十二、预期效果与调参建议
- 十三、与 DDPM 课程的联系

---

## 零、10 节学习路线总览

```
06.1  翻译是什么          （纯概念，无神经网络）
  ↓
06.2  文字 → 数字         （分词、词表、Batch）
  ↓
06.3  注意力              （Q/K/V、softmax 加权）
  ↓
06.4  位置编码            （词序信息）
  ↓
06.5  多头 + Mask         （Pad / Causal）
  ↓
06.6  Cross-Attention    （译时查原文、源句对齐）
  ↓
06.7  拼成 Transformer    （Encoder-Decoder、维度检查）
  ↓
06.8  训练                （Teacher Forcing、CrossEntropy）
  ↓
06.9  推理                （贪心解码、交互翻译）
  ↓
06.10 可视化 Attention   （真正画出模型对齐热力图）
```

| 小节 | 脚本 | 核心问题 | 预计耗时 |
|------|------|----------|----------|
| 06.1 | `01_seq2seq_intro.py` | 翻译和查字典有什么不同？ | 1 分钟 |
| 06.2 | `02_data_vocab.py` | 字符串怎么喂给神经网络？ | 1 分钟 |
| 06.3 | `03_attention_intuition.py` | 译某个词时该「看」原文哪里？ | 1 分钟 |
| 06.4 | `04_positional_encoding.py` | 模型怎么知道词序？ | 1 分钟 |
| 06.5 | `05_multihead_and_mask.py` | 为什么要 Mask？多头自注意力？ | 1 分钟 |
| 06.6 | `06_cross_attention.py` | Cross-Attn 怎么连接 Encoder 与 Decoder？ | 2 分钟 |
| 06.7 | `07_model_assembly.py` | 完整模型长什么样？shape 怎么变？ | 1 分钟 |
| 06.8 | `08_train.py` | 怎么让模型学会翻译？ | CPU 约 1 分钟（`--fast`） |
| 06.9 | `09_infer.py` | 只有英文时怎么生成中文？ | 1 分钟 |
| 06.10 | `10_visualize_attention.py` | 训练后的模型到底在「看」哪里？ | 1 分钟 |

**建议**：按顺序学习，06.1~06.7 只读 + 跑脚本即可；06.8 需要等前面理解后再跑；06.10 在有 checkpoint 后跑。

```bash
conda activate ddpm_learn
cd section06_transformer_mt

python 01_seq2seq_intro.py
python 02_data_vocab.py
# … 依次到 07 …
python 08_train.py --fast
python 09_infer.py --sentence "i love you"
python 10_visualize_attention.py --sentence "i love you"
```

---

## 一、06.1 机器翻译在做什么？（Seq2Seq 直觉）

### 1.1 我们要解决什么问题

```
输入：一句英文    "i love you"
输出：一句中文    "我爱你"
```

这看起来简单，但机器翻译 **不是** 逐词查字典：

| 英文 | 若逐词翻译 | 正确译文 | 问题 |
|------|-----------|----------|------|
| good morning | 好 + 早上 | 早上好 | 词数不对（2→3） |
| where is the book | 哪里 + 是 + 这 + 书 | 书在哪里 | 语序不对 |
| i am a student | 我 + 是 + 一个 + 学生 | 我是学生 | 「a」中文里常省略 |

所以需要模型理解 **整句语义**，再 **生成整句译文**。输入、输出都是长度可变的序列，这类任务叫 **Seq2Seq**（Sequence to Sequence，序列到序列）。

### 1.2 Encoder-Decoder：先「读懂」，再「写出」

Transformer 采用 **Encoder-Decoder** 结构，可以先用生活类比理解：

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  英文原句     │  ──►   │   Encoder    │  ──►   │   memory     │
│ i love you   │         │  （读原文）   │         │ （句向量）    │
└──────────────┘         └──────────────┘         └──────┬───────┘
                                                         │
┌──────────────┐         ┌──────────────┐                │
│  中文译文     │  ◄──   │   Decoder    │  ◄─────────────┘
│ 我爱你       │         │ （写译文）    │
└──────────────┘         └──────────────┘
```

| 组件 | 作用 | 类比 |
|------|------|------|
| **Encoder** | 读完整英文，压缩成语义表示 `memory` | 先通读原文，理解大意 |
| **Decoder** | 根据 `memory`，一个字一个字写中文 | 看着原文笔记，逐字写译文 |
| **memory** | Encoder 的输出，供 Decoder 随时查阅 | 对原文的「压缩笔记」 |

### 1.3 训练 vs 推理：有什么不一样？

| | 训练 | 推理 |
|--|------|------|
| 输入 | 英文 + **标准中文**（对照句） | **只有英文** |
| Decoder 怎么写 | 有标准答案，用 Teacher Forcing 学 | 从 `<bos>` 开始，自己猜下一个字 |
| 停止条件 | 已知完整句子长度 | 生成到 `<eos>` 或达到最大长度 |

训练时我们有很多 `(英文, 中文)` 对照对；推理时用户只给英文，模型必须 **自回归**（autoregressive）地逐 token 生成。

### 1.4 本仓库的数据

`data_utils.py` 内置约 **80 对** 英→中短句，**无需联网**：

```
i love you          →  我爱你
good morning        →  早上好
where is the book   →  书在哪里
…
```

教学用的小语料，目的是 **跑通流程**，不是做工业级翻译。

### 1.5 运行方式

```bash
python 01_seq2seq_intro.py
```

脚本会打印：问题说明、Encoder-Decoder 图示、前 5 对语料、后续小节指引。**本节不涉及任何 PyTorch 代码。**

### 1.6 本节小结

- 机器翻译 = Seq2Seq，输入输出都是序列
- Encoder 读原文 → memory；Decoder 看 memory 写译文
- 训练有对照句；推理只能逐字生成
- 下一节：神经网络只吃数字，句子怎么变成数字？

---

## 二、06.2 从文字到数字（分词与词表）

### 2.1 为什么需要这一步？

神经网络的基本运算是矩阵乘法，只能处理 **数字**。因此必须：

```
"i love you"  →  [1, 4, 36, 8, 2]   （整数 id 序列）
```

整个过程：**分词 → 查词表 → 编码**；反过来 **解码 → 还原文字**。

### 2.2 分词（Tokenization）

本项目采用最简单的规则（教学够用）：

| 语言 | 规则 | 示例 |
|------|------|------|
| 英文 | 转小写，按空格切 | `"I Love You"` → `["i", "love", "you"]` |
| 中文 | 按 **字** 切 | `"我爱你"` → `["我", "爱", "你"]` |

代码见 `data_utils.py`：

```python
def tokenize_en(text: str) -> list[str]:
    text = text.lower().strip()
    return text.split()

def tokenize_zh(text: str) -> list[str]:
    return list(text.strip())
```

> 工业界常用 BPE / SentencePiece 等子词分词；这里按字/词切是为了 **直观**，不引入额外依赖。

### 2.3 词表（Vocabulary）

词表 = 字典：`词/字 → 整数 id`。

四个 **特殊符号**（任何 Seq2Seq 几乎都有）：

| 符号 | 全称 | id 通常 | 作用 |
|------|------|---------|------|
| `<pad>` | padding | 0 | 补齐短句，凑成同一长度 |
| `<bos>` | begin of sequence | 1 | 告诉 Decoder「开始写」 |
| `<eos>` | end of sequence | 2 | 告诉模型「写完了」 |
| `<unk>` | unknown | 3 | 词表里没有的词 |

编码格式（每句首尾加 bos/eos）：

```
英文 "i love you"
  tokens:  [i, love, you]
  ids:     [<bos>, i, love, you, <eos>]
           [  1,   4,   36,   8,    2]

中文 "我爱你"
  tokens:  [我, 爱, 你]
  ids:     [<bos>, 我, 爱, 你, <eos>]
```

### 2.4 Batch：多句话怎么拼在一起？

一个 batch 里各句长度不同，需要 **pad 到同一长度**：

```
句子 A: [bos, i, love, you, eos]           长度 5
句子 B: [bos, i, like, cats, eos]          长度 5
句子 C: [bos, he, is, a, teacher, eos]     长度 6  ← batch 内最长

pad 后矩阵 src（shape = batch_size × max_len）:
  A: [1, 4, 36,  8,  2]
  B: [1, 4, 12, 24,  2]
  C: [1, 5,  9,  3, 15, 2]
```

`<pad>` 位置的 loss 和 attention 都会被 **忽略**（06.5 讲 Mask，06.8 训练里 `ignore_index=pad`）。

### 2.5 运行方式

```bash
python 02_data_vocab.py
```

你会看到：分词结果、词表大小、encode/decode 往返、一个 mini-batch 的 shape。

### 2.6 常见疑问

**Q：为什么中文按字而不是按词？**  
A：中文没有天然空格，按字切最简单；80 句短语料里几乎无未登录词问题。

**Q：`<bos>` 和 `<eos>` 能否省略？**  
A：不行。Decoder 需要 `<bos>` 作为生成的起点；`<eos>` 告诉推理何时停止。

### 2.7 本节小结

- 分词 → 词表 id → 矩阵，是进入模型的必经之路
- 特殊符号：`<pad>` 对齐、`<bos>` 开始、`<eos>` 结束
- 下一节：数字矩阵进网络后，模型怎么「关注」重要信息？→ **注意力**

---

## 三、06.3 注意力机制直觉

### 3.1 动机：翻译时「看哪里」？

译「爱」这个字时，模型应该重点看英文里的 `love`；译「你」时看 `you`。  
**Attention（注意力）** 就是：当前位置对序列里各位置分配 **权重**，权重大的地方信息被多吸收。

### 3.2 三个向量：Q、K、V

| 符号 | 名称 | 直觉 |
|------|------|------|
| **Q** | Query（查询） | 「我现在想找什么信息？」 |
| **K** | Key（键） | 「每个位置提供什么标签/索引？」 |
| **V** | Value（值） | 「每个位置实际携带的内容」 |

Self-Attention 时 Q、K、V 都来自 **同一句** 的表示（只是经不同线性层投影）。

### 3.3 缩放点积注意力：三步

**Step 1 — 算相似度**

$$\text{scores} = \frac{QK^\top}{\sqrt{d_k}}$$

- `scores[i, j]` = 第 $i$ 个位置对第 $j$ 个位置的关注程度（未归一化）
- 除以 $\sqrt{d_k}$：**缩放**，防止维度大时点积过大，softmax 变成「近似 one-hot」

**Step 2 — Softmax 归一化**

$$\text{weights} = \mathrm{softmax}(\text{scores}) \quad \text{（每行和为 1）}$$

**Step 3 — 加权求和 Value**

$$\text{output} = \text{weights} \cdot V$$

每个位置输出一个 **融合了全局上下文** 的新向量。

完整公式：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

### 3.4 手工小例子

脚本 `03_attention_intuition.py` 用 3 个词 `["i", "love", "you"]`、4 维随机向量演示全流程，并打印：

```
scores 矩阵  →  softmax 权重  →  加权输出 shape (3, 4)
```

同时保存热力图 `figures/03_attention_heatmap.png`：行 = 谁在看，列 = 被看谁。

### 3.5 Self-Attention vs Cross-Attention

| 类型 | Q 来自 | K、V 来自 | 用途 |
|------|--------|-----------|------|
| **Self-Attention** | 本句 | 本句 | 理解句内上下文 |
| **Cross-Attention** | Decoder（译文） | Encoder（原文 memory） | 译时「查原文」 |

Cross-Attention 是机器翻译的关键：写中文每个字时，都可以 **回头查** Encoder 对英文的理解。

### 3.6 运行方式

```bash
python 03_attention_intuition.py
```

### 3.7 本节小结

- Attention = 相似度 → softmax 权重 → 加权求和
- Q/K/V 分工：查询、索引、内容
- Cross-Attention 是机器翻译的关键 → **06.6 专节详解**
- 下一节：Attention 本身 **不区分顺序**，怎么告诉模型词序？

---

## 四、06.4 位置编码

### 4.1 问题：Attention 是「集合运算」

如果只把词变成向量再做 Self-Attention，**打乱顺序结果不变**：

```
"i love you"  和  "you love i"  →  得到相同的 attention 输出（集合相同）
```

但语序对翻译至关重要，必须显式注入 **位置信息**。

### 4.2 做法：Embedding + 位置编码

$$\text{输入向量} = \text{Embedding}(\text{词}) + \text{PE}(\text{位置})$$

- Embedding：查表得到词的语义向量（可学习参数）
- PE：按位置算出来的固定（或可变）向量

Transformer 原文用 **正弦位置编码**（Sinusoidal PE），与 Section 04 DDPM 的 **时间嵌入** 公式同源：

$$PE_{(pos,\,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,\,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

| 符号 | 含义 |
|------|------|
| $pos$ | 词在句中的位置（0, 1, 2, …） |
| $i$ | 维度索引 |
| $d_{\text{model}}$ | 模型隐层维度（如 128） |

不同维度用不同频率的正弦/余弦，使每个位置都有 **唯一** 的编码模式。

### 4.3 与 Section 04 的对照

| | Section 04 (DDPM) | Section 06 (Transformer) |
|--|-------------------|--------------------------|
| 输入标量 | 扩散时间步 $t$ | 词位置 $pos$ |
| 编码方式 | 正弦/余弦 | 正弦/余弦 |
| 作用 | 告诉 U-Net 当前去噪阶段 | 告诉模型当前是第几个词 |
| 代码 | `SinusoidalPositionEmbeddings` | `PositionalEncoding` |

你在 DDPM 里已经见过这套编码；这里只是 **把「时间」换成「位置」**。

### 4.4 代码要点

`transformer.py` 中：

```python
class PositionalEncoding(nn.Module):
    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[:, :x.size(1)]  # 按位置加上 PE
        return self.dropout(x)
```

进入 Encoder/Decoder 前，还有一步：`embedding * sqrt(d_model)`，是论文里的缩放技巧，使两项量级接近。

### 4.5 运行方式

```bash
python 04_positional_encoding.py
```

输出前几个位置的 PE 数值，并保存 `figures/04_positional_encoding.png`（位置 × 维度 热力图）。

### 4.6 本节小结

- 没有位置编码，模型分不清词序
- PE 与 DDPM 时间嵌入公式相同，只是语义不同
- 下一节：单头不够 → 多头；还要 Mask 掉 pad 和未来

---

## 五、06.5 多头注意力与 Mask

### 5.1 多头注意力（Multi-Head）

一组 Q/K/V 只能学一种「关注模式」。**多头** = 并行开 $h$ 组独立的注意力，最后拼接：

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)\,W^O$$

$$\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,\,KW_i^K,\,VW_i^V)$$

| 参数 | 本项目默认 | 含义 |
|------|-----------|------|
| `d_model` | 128 | 总隐层维度 |
| `n_heads` | 4 | 头数 |
| `d_k = d_model / n_heads` | 32 | 每个头的维度 |

直觉：不同头可能分别关注 **语法结构、指代关系、词序邻近** 等，最后拼起来信息更丰富。

### 5.2 Pad Mask：忽略补齐位

Batch 里短句末尾是 `<pad>`，不能让模型 attend 到 pad，也不能对 pad 算 loss。

```
ids:     [1,  4, 36,  8,  2,  0,  0]
         bos  i love you eos pad pad

pad_mask（1=有效，0=pad）:
         [1,  1,  1,  1,  1,  0,  0]
```

实现：`make_pad_mask(seq, pad_idx)` → shape `(B, 1, 1, L)`，广播到 attention scores。

### 5.3 Causal Mask：Decoder 不能看未来

训练 Decoder 时，预测第 $t$ 个 token **只能看到 $t$ 之前** 的内容，不能偷看标准答案后面的字：

```
        可看列 j →
      j=0  j=1  j=2  j=3
i=0    1    0    0    0
i=1    1    1    0    0
i=2    1    1    1    0
i=3    1    1    1    1
（下三角矩阵）
```

这叫 **因果 Mask**（Causal / Look-ahead Mask）。  
Decoder 实际用的是：**Pad Mask ∩ Causal Mask**（`make_tgt_mask`）。

### 5.4 三种注意力在本项目中的分工（Cross-Attn 详见 06.6）

```
Encoder:
  英文 ──► Self-Attn（Q,K,V 都来自英文，Pad Mask）

Decoder:
  中文 ──► Masked Self-Attn（Q,K,V 来自已生成中文，Pad + Causal）
       ──► Cross-Attn（Q 来自中文，K,V 来自 memory）  ← 专节 06.6
       ──► FFN
```

| 模块 | Q | K | V | Mask |
|------|---|---|---|------|
| Encoder Self-Attn | 英文 | 英文 | 英文 | Pad |
| Decoder Self-Attn | 中文 | 中文 | 中文 | Pad + Causal |
| Cross-Attn | 中文 | memory | memory | Pad（源句） |

### 5.5 运行方式

```bash
python 05_multihead_and_mask.py
```

打印多头输出 shape、pad mask 示例、因果 mask 下三角矩阵。

### 5.6 本节小结

- 多头 = 多组注意力并行，再拼接
- Pad Mask：对齐用的空位不参与计算
- Causal Mask：Decoder 预测时不能看未来
- Cross-Attn 是 Encoder-Decoder 的桥梁 → **下一节专讲 06.6**

---

## 六、06.6 Cross-Attention（交叉注意力）专节

### 6.1 动机：写译文时要「查原文」

06.3 学了注意力公式，06.5 学了多头和 Mask。机器翻译里最关键的一种注意力是 **Cross-Attention**：

- Encoder 读完英文，输出 `memory`：每个英文位置一个向量（不是压成一个向量）
- Decoder 每写一个中文字，都要问：**原文里哪个位置和我现在要写的内容最相关？**

这就是 Cross-Attention 的作用——**软对齐**（soft alignment）。

### 6.2 与 Self-Attention 的唯一区别：Q/K/V 从哪来

公式与 06.3 **完全相同**：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

| | Self-Attention | Cross-Attention |
|--|----------------|-----------------|
| Q | 本句 | **Decoder（译文）** |
| K | 本句 | **Encoder memory（原文）** |
| V | 本句 | **Encoder memory** |
| 对齐矩阵 shape | $(L, L)$ 方阵 | $(L_{\text{tgt}}, L_{\text{src}})$ 矩形 |

### 6.3 翻译小例子：`i love you` → `我爱你`

脚本 `06_cross_attention.py` 用手工向量演示对齐矩阵（行=中文 Query，列=英文 Key）：

```
         i    love   you
<bos>  0.xx  0.xx  0.xx
我     高    低    低     ← 写「我」时主要看 i
爱     低    高    低     ← 写「爱」时主要看 love
你     低    低    高     ← 写「你」时主要看 you
```

真实训练里这些权重是 **学出来** 的，不是手写规则；但可视化后往往呈现类似模式。

### 6.4 Decoder 一层内的顺序

```
x ──► Masked Self-Attn（中文内部互相看，+ Causal）
  ──► Cross-Attn（x 作 Q，memory 作 K/V）  ← 本节重点
  ──► FFN
```

- **Self-Attn 先**：让已生成的中文内部保持语序与连贯
- **Cross-Attn 后**：让每个中文位置去英文里「查资料」

代码（`transformer.py` 的 `DecoderLayer`）：

```python
x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, memory_mask)))
```

注意：`cross_attn` 与 `self_attn` 用的是 **同一个** `MultiHeadAttention` 类，只是 `query/key/value` 参数不同。

### 6.5 Cross-Attn 的 Mask

- **需要**：源句 **Pad Mask**（英文 `<pad>` 列权重为 0）
- **不需要**：Causal Mask（写「你」时可以 attend 到英文任意有效位置，包括后面的 `you`）

Causal 只约束 Decoder **Self-Attn**（不能偷看未来中文），不约束查原文。

### 6.6 比旧 RNN Seq2Seq 强在哪？

| | RNN Encoder-Decoder | Transformer Cross-Attn |
|--|---------------------|------------------------|
| 原文表示 | 整句压成 **一个** 向量 | 保留 **每个位置** 的向量 |
| 对齐 | 隐式、难解释 | 显式 $(L_{\text{tgt}} \times L_{\text{src}})$ 矩阵，可热力图 |
| 长句 | 信息易丢失 | 每步可「点名」查任意英文位置 |

### 6.7 运行方式

```bash
python 06_cross_attention.py
```

输出：Self vs Cross 对比、shape 追踪、Pad Mask 演示，并保存 `figures/06_cross_attention_heatmap.png`。

### 6.8 本节小结

- Cross-Attn = 同一公式，Q 来自 Decoder，K/V 来自 Encoder memory
- 对齐矩阵行=译文位置，列=原文位置
- 在 Decoder 里位于 Masked Self-Attn 之后、FFN 之前
- 下一节：把 Encoder、Decoder、Cross-Attn **拼成完整模型**

---

## 七、06.7 组装完整 Transformer

### 7.1 整体架构

```
                    ┌─── Encoder × N 层 ───┐
src ids ──► Embed+PE ──► Self-Attn ──► FFN ──► … ──► memory
                                                      │
                    ┌─── Decoder × N 层 ───┐          │
tgt ids ──► Embed+PE ──► Masked Self-Attn ────────────┤
                      ──► Cross-Attn ◄── memory ──────┘
                      ──► FFN ──► … ──► Linear ──► logits（词表大小）
```

### 7.2 单层 Encoder

```
输入 x
  → Self-Attention(x, x, x) + 残差
  → LayerNorm
  → FFN（两层 Linear + ReLU）+ 残差
  → LayerNorm
  → 输出
```

### 7.3 单层 Decoder（比 Encoder 多 Cross-Attn）

```
输入 x，以及 Encoder 的 memory
  → Masked Self-Attention + 残差 + LayerNorm
  → Cross-Attention(x, memory, memory) + 残差 + LayerNorm
  → FFN + 残差 + LayerNorm
  → 输出
```

### 7.4 Teacher Forcing：训练时输入/标签错位一行

训练时我们有完整中文，但 **不能** 把整句中文一次性塞给 Decoder 让它「抄答案」。  
正确做法：**输入比标签右移一位**——每个位置只预测 **下一个** token。

```
完整 tgt:  [<bos>,  我,   爱,   你,  <eos>]
tgt_in:     [<bos>,  我,   爱,   你]          ← Decoder 输入
tgt_out:    [ 我,    爱,   你,  <eos>]       ← 要预测的目标（标签）

位置 0：看到 <bos>        → 预测「我」
位置 1：看到 <bos> 我     → 预测「爱」
位置 2：看到 <bos> 我 爱  → 预测「你」
…
```

代码（`08_train.py`）：

```python
tgt_in  = tgt[:, :-1]   # 去掉最后一个
tgt_out = tgt[:, 1:]    # 去掉 <bos>
logits  = model(src, tgt_in, src_mask, tgt_mask)
loss    = CrossEntropy(logits, tgt_out, ignore_index=pad)
```

### 7.5 维度追踪（以 `"i love you" → "我爱你"` 为例）

| 张量 | shape | 含义 |
|------|-------|------|
| `src` | `(B, L_src)` | 英文 id |
| `tgt_in` | `(B, L_tgt-1)` | Decoder 输入 |
| `memory` | `(B, L_src, d_model)` | Encoder 输出 |
| `logits` | `(B, L_tgt-1, V_tgt)` | 每个位置对目标词表各类的得分 |
| `V_tgt` | 标量 | 目标词表大小 |

### 7.6 运行方式

```bash
python 07_model_assembly.py
```

用小模型跑一遍前向传播，打印上述 shape，**确认维度无误再训练**。

### 7.7 本节小结

- Encoder 产出 memory；Decoder 读 memory 写 logits
- Teacher Forcing：`tgt_in` / `tgt_out` 错一位
- 模型定义在 `transformer.py`，本节只做 **组装与验维度**
- 下一节：用 80 对句子 **真正训练**

---

## 八、06.8 训练（Teacher Forcing）

### 8.1 训练在优化什么？

对每个位置，模型输出词表上每个 token 的得分 `logits`，与真实下一个 token 做 **交叉熵**：

$$L = -\sum \log p(\text{真实 token} \mid \text{上下文})$$

`<pad>` 位置不参与（`ignore_index=0`）。

### 8.2 一步训练流程

```
1. 从 DataLoader 取 batch：src, tgt
2. tgt_in = tgt[:, :-1],  tgt_out = tgt[:, 1:]
3. 构造 src_mask, tgt_mask
4. logits = model(src, tgt_in, src_mask, tgt_mask)
5. loss = CrossEntropy(logits, tgt_out)
6. loss.backward() → 梯度裁剪 → optimizer.step()
```

### 8.3 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 80 | 训练轮数 |
| `--batch-size` | 32 | 批大小 |
| `--lr` | 3e-4 | AdamW 学习率 |
| `--d-model` | 128 | 隐层维度 |
| `--n-heads` | 4 | 注意力头数 |
| `--n-layers` | 2 | Encoder / Decoder 各 2 层 |
| `--d-ff` | 256 | FFN 中间层维度 |
| `--dropout` | 0.1 | Dropout |
| `--fast` | - | 120 epoch + 略降 dropout，CPU 约 1 分钟 |

优化器：**AdamW**（`betas=(0.9, 0.98)`, `weight_decay=1e-4`），梯度裁剪 `max_norm=1.0`。

### 8.4 运行方式

```bash
# 快速演示（推荐第一次跑）
python 08_train.py --fast

# 完整训练
python 08_train.py --epochs 80

# 自定义平行语料（TSV：src\ttgt 每行一对）
python 08_train.py --data tsv --tsv my_pairs.tsv
```

### 8.5 输出文件

| 路径 | 内容 |
|------|------|
| `checkpoints/last.pth` | **最后一轮**权重（小语料推理推荐用这个） |
| `checkpoints/best.pth` | 验证 loss 最低的一轮 |
| `checkpoints/src_vocab.json` | 源词表 |
| `checkpoints/tgt_vocab.json` | 目标词表 |
| `checkpoints/config.json` | 模型结构参数 |
| `figures/loss_curve.png` | train / val 损失曲线 |

> 内置语料只有 ~80 句，验证集 ~8 句，`best.pth` 可能 **过早停止**；充分过拟合后 `last.pth` 翻译效果更好。

### 8.6 训练过程预期

| Epoch | train loss | 翻译样例（大致） |
|-------|------------|-----------------|
| 1~5 | ~4~5 | 乱码、重复字 |
| 20~40 | ~1~2 | 部分短句可读 |
| 120 (`--fast`) | ~0.03 | 「我爱你」「书在哪里」等基本正确 |

每 `--eval-every` 个 epoch 会打印 5 句 demo 翻译。

### 8.7 常见问题

| 现象 | 可能原因 |
|------|----------|
| loss 不下降 | `tgt_in`/`tgt_out` 没错位；或 `ignore_index` 没设 pad |
| 输出全是同一个字 | 训练不充分；或学习率过大 |
| val loss 升、train loss 降 | 小语料过拟合，正常；推理用 `last.pth` |

### 8.8 本节小结

- 损失 = 逐 token 交叉熵，Teacher Forcing 提供上下文
- `--fast` 即可在 CPU 看到像样结果
- 训练完 → `09_infer.py` 做真正「只有英文」的翻译

---

## 九、06.9 推理（自回归解码）

### 9.1 训练和推理的核心区别

| | 训练 | 推理 |
|--|------|------|
| 中文从哪来 | DataLoader 提供完整 `tgt` | **没有**，只能自己生成 |
| Decoder 输入 | 真实的 `tgt_in`（Teacher Forcing） | 上一步 **自己预测** 的字 |
| 停止 | 已知句子长度 | 遇到 `<eos>` 或达到 `max_len` |

### 9.2 贪心解码（Greedy Decode）

本项目采用最基础的策略：**每步取概率最大的 token**。

```python
memory = Encoder(src)           # 原文只编码一次
ys = [<bos>]                    # 从 begin 开始

for step in range(max_len):
    logits = Decoder(ys, memory)[:, -1, :]   # 最后一个位置的预测
    next_token = argmax(logits)
    ys.append(next_token)
    if next_token == <eos>:
        break

return decode(ys)               # id → 中文文本
```

代码见 `transformer.py` 的 `greedy_decode()`。

### 9.3 为什么不用 Teacher Forcing 推理？

推理时没有标准译文。若把整句 gold 中文塞进去，等于 **抄答案**，无法评估真实翻译能力。  
必须 **自回归**：第 1 步预测的字，成为第 2 步的输入，误差会累积（Exposure Bias，进阶话题）。

### 9.4 运行方式

```bash
# 需先训练
python 08_train.py --fast

# 单句翻译
python 09_infer.py --sentence "i love you"

# 交互模式（先演示 5 句，再 EN> 输入）
python 09_infer.py

# 指定 checkpoint
python 09_infer.py --checkpoint checkpoints/last.pth
```

### 9.5 预期输出示例

```
EN: i love you
ZH: 我爱你

EN: where is the book
ZH: 书在哪里

EN: i am a student
ZH: 我是学生
```

未见过的长句、复杂句式可能出错——内置语料太小，属正常。

### 9.6 进阶方向（本仓库未实现）

- **Beam Search**：每步保留 top-k 候选，质量更好、更慢
- **BPE 分词 + 大数据**（WMT / Multi30k）：才有泛化能力
- **Label Smoothing、Warmup**：稳定训练

### 9.7 本节小结

- 推理 = Encoder 一次 + Decoder 逐步贪心生成
- 默认加载 `checkpoints/last.pth`
- 下一节：把模型内部的 **Cross-Attention 权重画出来**，确认它真的在「对齐全」

---

## 十、06.10 可视化 Attention（看模型「看了哪里」）

### 10.1 为什么以前「看不见」？

默认的 `MultiHeadAttention.forward` 只返回融合后的向量，权重矩阵在 softmax 之后马上就丢掉了。  
训练只优化 loss，控制台也只打印句子——所以你会觉得 **TransformerMT 里没有 attention**。

现在模型增加了：

```python
logits, attns = model.forward_with_attention(src, tgt_in, src_mask, tgt_mask)
# attns["cross"][layer] : (B, n_heads, L_tgt, L_src)
```

### 10.2 怎么用

```bash
# 需先训练出 checkpoint
python 08_train.py --fast

# 画最后一层、各头平均的 Cross-Attn（最推荐）
python 10_visualize_attention.py --sentence "i love you"

# 看某一个头 / 某一层
python 10_visualize_attention.py --sentence "where is the book" --layer 0 --head 2

# 把所有头画在一张图
python 10_visualize_attention.py --sentence "i love you" --show-all-heads

# 推理时顺带开可视化
python 09_infer.py --sentence "i love you" --viz
```

热力图保存在 `figures/10_cross_attn_*.png`。  
另外，`08_train.py` 每 `--eval-every` 轮也会自动存 `figures/attn_train_epochXXX_i_love_you.png`，训练过程中就能对比「早期乱对齐 → 后期靠近对角」。

### 10.3 怎么读图

| 轴 | 含义 |
|----|------|
| 行 | 正在写的中文字（Query） |
| 列 | 英文位置（Key = Encoder memory） |
| 颜色深 | 写这个字时更关注该英文词 |

理想情况（训好后）：「我」对 `i`、「爱」对 `love`、「你」对 `you` 权重更高——这就是学到的 **软对齐**。

### 10.4 本节小结

- Attention 权重本来就算过，只是以前没返回、没画
- `forward_with_attention` + `10_visualize_attention.py` 让学习可检验
- Section 06 闭环：**懂问题 → 训出模型 → 看见它对齐了哪里**

---

## 十一、文件说明与环境

### 11.1 目录结构

| 文件 / 目录 | 对应小节 | 说明 |
|-------------|----------|------|
| `01_seq2seq_intro.py` | 06.1 | Seq2Seq 直觉演示 |
| `02_data_vocab.py` | 06.2 | 分词、词表、Batch |
| `03_attention_intuition.py` | 06.3 | 注意力 + 热力图 |
| `04_positional_encoding.py` | 06.4 | 位置编码可视化 |
| `05_multihead_and_mask.py` | 06.5 | 多头与 Mask |
| `06_cross_attention.py` | 06.6 | Cross-Attention 专节 + 对齐热力图 |
| `07_model_assembly.py` | 06.7 | 模型组装、维度检查 |
| `08_train.py` | 06.8 | 训练主脚本（eval 时存 attn 图） |
| `09_infer.py` | 06.9 | 推理 / 交互翻译（`--viz` 可画 attn） |
| `10_visualize_attention.py` | 06.10 | 训练后 Attention 可视化 |
| `transformer.py` | 06.4~06.10 | 模型、`forward_with_attention`、`greedy_decode` |
| `data_utils.py` | 06.2~06.8 | 语料、词表、Dataset |
| `checkpoints/` | 06.8 | 权重（gitignore） |
| `figures/` | 06.3~06.10 | 可视化图 |

### 11.2 环境

```bash
conda activate ddpm_learn
cd section06_transformer_mt
```

依赖：`torch`, `matplotlib`, `tqdm`（与 Section 05 相同，无需 diffusers）。

### 11.3 一键顺序跑通

```bash
python 01_seq2seq_intro.py
python 02_data_vocab.py
python 03_attention_intuition.py
python 04_positional_encoding.py
python 05_multihead_and_mask.py
python 06_cross_attention.py
python 07_model_assembly.py
python 08_train.py --fast
python 09_infer.py --sentence "i love you"
python 10_visualize_attention.py --sentence "i love you"
```

---

## 十二、预期效果与调参建议

### 12.1 各阶段目标

| 阶段 | 你应该能回答 |
|------|-------------|
| 06.1 | 翻译为什么不是查字典？Encoder/Decoder 各干什么？ |
| 06.2 | `<bos>/<eos>/<pad>` 是什么？Batch 怎么 pad？ |
| 06.3 | Q/K/V 是什么？Attention 三步公式？ |
| 06.4 | 为什么需要位置编码？与 DDPM 时间嵌入有何关系？ |
| 06.5 | Pad / Causal Mask 分别解决什么问题？ |
| 06.6 | Cross-Attn 的 Q/K/V 从哪来？对齐矩阵 shape？ |
| 06.7 | Teacher Forcing 如何错位？logits shape 是什么？ |
| 06.8 | 训练 loss 是什么？checkpoint 存在哪？ |
| 06.9 | 推理和训练有何不同？贪心解码怎么做？ |
| 06.10 | Cross-Attn 热力图行列各代表什么？怎样才算对齐学好了？ |

### 12.2 调参建议

- **只想验证流程**：`08_train.py --fast`，CPU 即可
- **想看见 attention**：训完后跑 `10_visualize_attention.py`；训练中看 `figures/attn_train_*.png`
- **翻译仍很差**：增加 `--epochs`；小语料用 `last.pth` 而非 `best.pth`
- **想换数据**：准备 `en\tszh` 的 TSV，`--data tsv --tsv path`
- **想更大模型**：提高 `--d-model`、`--n-layers`（需更多数据，否则更易过拟合）

### 12.3 重要提醒

> 内置 ~80 句 **仅供学习 Transformer 流程**，不是可用的翻译产品。  
> 真实系统需要 **百万级句对 + 子词分词 + 更大模型 + Beam Search**。

---

## 十三、与 DDPM 课程的联系

| 概念 | DDPM (Section 01~05) | Transformer (Section 06) |
|------|----------------------|--------------------------|
| 任务 | 生成图像 | 生成文本（翻译） |
| 输入条件 | 时间步 $t$ | 源句 + 已生成部分译文 |
| 序列性 | 1000 步去噪 | 逐 token 生成 |
| 正弦编码 | 时间嵌入（Section 04） | 位置编码（06.4） |
| 注意力 | U-Net 瓶颈 Self-Attn | Self-Attn + Cross-Attn |
| 预测目标 | 连续噪声 $\varepsilon$ | 离散 token 分布 |
| 采样 | 反向去噪 $x_T\to x_0$ | 自回归 $<\bos>\to<\eos>$ |

---

## 课程衔接

| 章节 | 内容 |
|------|------|
| Section 01 | 正向加噪直觉与 Beta Schedule |
| Section 02 | 反向后验推导与采样公式 |
| Section 03 | ELBO → 简化噪声预测损失 |
| Section 04 | 时间嵌入 + U-Net 架构 |
| Section 05 | MNIST 训练 + 采样生成 |
| **Section 06** | **Transformer 机器翻译（10 小节循序渐进）** |

恭喜你完成从 DDPM 到 Transformer 的扩展学习路径。
