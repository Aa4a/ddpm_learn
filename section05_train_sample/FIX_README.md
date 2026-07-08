# Section 05 训练效果修复记录

## 问题现象

`--fast` 模式 2 epoch 训练后，采样图几乎全是噪声，无法辨认任何数字。

## 根因分析

| 问题 | 说明 |
|------|------|
| 模型容量不足 | SimpleUNet 通道数硬编码为 16/32，仅 157K 参数，无法学习有效去噪 |
| Batch size 过小 | 默认 128，RTX 2080 Ti 11GB 显存占用不到 5%，GPU 严重浪费 |
| time_dim 偏小 | 时间嵌入维度 128，对 1000 步扩散 Schedule 表达力不够 |

## 修复内容

### 1. SimpleUNet 通道可配置化

**文件**: `section04_unet/01_unet_components.py`

将硬编码通道数改为构造函数参数，默认值不变以兼容 Section 04 测试。

```python
# 修改前
def __init__(self, in_channels=3, out_channels=3, time_dim=128):

# 修改后
def __init__(self, in_channels=3, out_channels=3, time_dim=128,
             channel_1=16, channel_2=32):
```

编码器、解码器、跳跃连接的通道数全部由 `c1`、`c2` 变量驱动。

### 2. 扩大模型容量

**文件**: `section05_train_sample/ddpm.py` — `load_simple_unet()`

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `time_dim` | 128 | 256 |
| `channel_1` | 16 | 128 |
| `channel_2` | 32 | 256 |
| 模型参数量 | 157K | 8.11M |

### 3. 增大 batch size

**文件**: `section05_train_sample/01_train_mnist.py`

```python
# 修改前
p.add_argument("--batch-size", type=int, default=128)

# 修改后
p.add_argument("--batch-size", type=int, default=896)
```

显存占用从 ~0.5GB 提升到 10.8GB / 11.3GB（96%）。

### 4. 同步所有脚本

`02_sample.py`、`03_val_pred_compare.py`、`04_compare_similarity.py` 中的 `load_simple_unet` 调用全部同步为新参数，否则旧 checkpoint 加载会报维度不匹配。

## 效果对比

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 模型参数 | 157K | 8.11M |
| Batch size | 128 | 896 |
| 显存占用 | ~5% | 96% |
| 2 epoch loss | 0.306 | 0.066 |
| 20 epoch loss | — | 0.025 |
| 采样效果 | 纯噪声 | 清晰可辨认数字 |
| 训练耗时 (20 epoch) | — | 19.2 分钟 |

## 训练 Loss 曲线

| Epoch | 平均损失 |
|-------|----------|
| 1 | 0.1754 |
| 2 | 0.0662 |
| 5 | 0.0379 |
| 10 | 0.0303 |
| 15 | 0.0276 |
| 20 | 0.0254 |

## 涉及文件

- `section04_unet/01_unet_components.py` — SimpleUNet 通道可配置化
- `section05_train_sample/ddpm.py` — `load_simple_unet` 默认参数升级
- `section05_train_sample/01_train_mnist.py` — batch size 默认值
- `section05_train_sample/02_sample.py` — 模型加载参数同步
- `section05_train_sample/03_val_pred_compare.py` — 模型加载参数同步
- `section05_train_sample/04_compare_similarity.py` — 模型加载参数同步

## 复现方式

```bash
conda activate ddpm_learn
cd section05_train_sample
python 01_train_mnist.py --epochs 20 --batch-size 896
```

采样：

```bash
python 02_sample.py
```
