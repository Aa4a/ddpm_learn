# -*- coding: utf-8 -*-
"""
Section 07 - 数据与词表工具

MNIST + 数字名「文本」条件（教学版 Stable Diffusion 的 prompt）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent


def setup_stdio() -> None:
    """Windows GBK 控制台下避免 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

# 数字 → 英文名（当作 prompt）
DIGIT_NAMES = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
]

# 特殊 token
PAD_TOKEN = "<pad>"
UNCOND_TOKEN = "<uncond>"  # Classifier-Free Guidance 的无条件占位

SPECIAL_TOKENS = [PAD_TOKEN, UNCOND_TOKEN]
VOCAB = SPECIAL_TOKENS + DIGIT_NAMES
TOKEN2ID = {tok: i for i, tok in enumerate(VOCAB)}
ID2TOKEN = {i: tok for tok, i in TOKEN2ID.items()}
PAD_ID = TOKEN2ID[PAD_TOKEN]
UNCOND_ID = TOKEN2ID[UNCOND_TOKEN]
VOCAB_SIZE = len(VOCAB)
MAX_TEXT_LEN = 1  # 本教学版每个样本只有一个词（数字名）


def encode_prompt(text: str) -> list[int]:
    """将 prompt 编成 id 序列（小写、按空格切，只取词表内词）。"""
    text = text.strip().lower()
    if not text:
        return [UNCOND_ID]
    tokens = text.split()
    ids = [TOKEN2ID.get(tok, UNCOND_ID) for tok in tokens]
    return ids[:MAX_TEXT_LEN] if ids else [UNCOND_ID]


def decode_ids(ids: list[int] | torch.Tensor) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    words = [ID2TOKEN.get(int(i), "?") for i in ids if int(i) != PAD_ID]
    return " ".join(words)


def uncond_ids(batch_size: int = 1) -> torch.Tensor:
    """无条件文本 id，shape (B, MAX_TEXT_LEN)。"""
    return torch.full((batch_size, MAX_TEXT_LEN), UNCOND_ID, dtype=torch.long)


class MNISTTextDataset(Dataset):
    """MNIST 图像 + 对应数字名 token ids。"""

    def __init__(self, root: str | Path | None = None, train: bool = True):
        root = Path(root) if root is not None else HERE / "data"
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1]
        ])
        self.mnist = datasets.MNIST(
            root=str(root),
            train=train,
            download=True,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int):
        image, label = self.mnist[idx]
        text_ids = torch.tensor([TOKEN2ID[DIGIT_NAMES[label]]], dtype=torch.long)
        return image, text_ids, label


def collate_with_cfg_drop(
    batch,
    drop_prob: float = 0.1,
):
    """
    组装 batch；以 drop_prob 概率把文本替换为 <uncond>（CFG 训练）。
    返回: images (B,1,28,28), text_ids (B, L), labels (B,)
    """
    images, text_ids_list, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    text_ids = torch.stack(text_ids_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    if drop_prob > 0:
        drop_mask = torch.rand(len(batch)) < drop_prob
        text_ids = text_ids.clone()
        text_ids[drop_mask] = UNCOND_ID
    return images, text_ids, labels


def get_dataloader(
    batch_size: int,
    fast: bool = False,
    drop_prob: float = 0.1,
    train: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = MNISTTextDataset(train=train)
    if fast:
        n = min(5000, len(ds))
        ds = Subset(ds, range(n))

    def _collate(batch):
        return collate_with_cfg_drop(batch, drop_prob=drop_prob if train else 0.0)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
