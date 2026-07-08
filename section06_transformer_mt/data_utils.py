# -*- coding: utf-8 -*-
"""
Section 06 - 平行语料与词表

默认使用内置中英教学例句（无需联网），也可加载 Multi30k 或自定义 TSV。
词表：按空格分词（英文）/ 按字分词（中文），构建 <pad>/<bos>/<eos>/<unk>。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


# ---------------------------------------------------------------------------
# 内置平行语料（教学用，约 80 句，可快速在 CPU 上收敛出可读翻译）
# ---------------------------------------------------------------------------

BUILTIN_PAIRS: list[tuple[str, str]] = [
    ("i love you", "我爱你"),
    ("i like cats", "我喜欢猫"),
    ("i like dogs", "我喜欢狗"),
    ("he likes music", "他喜欢音乐"),
    ("she likes books", "她喜欢书"),
    ("we love china", "我们爱中国"),
    ("they like tea", "他们喜欢茶"),
    ("i am a student", "我是学生"),
    ("he is a teacher", "他是老师"),
    ("she is a doctor", "她是医生"),
    ("we are friends", "我们是朋友"),
    ("you are kind", "你很善良"),
    ("this is a book", "这是一本书"),
    ("that is a cat", "那是一只猫"),
    ("where is the book", "书在哪里"),
    ("where is the school", "学校在哪里"),
    ("what is your name", "你叫什么名字"),
    ("my name is tom", "我叫汤姆"),
    ("how are you", "你好吗"),
    ("i am fine", "我很好"),
    ("thank you", "谢谢你"),
    ("you are welcome", "不客气"),
    ("good morning", "早上好"),
    ("good night", "晚安"),
    ("see you tomorrow", "明天见"),
    ("i go to school", "我去学校"),
    ("he goes to work", "他去工作"),
    ("she goes home", "她回家"),
    ("we go to the park", "我们去公园"),
    ("i eat an apple", "我吃一个苹果"),
    ("he drinks water", "他喝水"),
    ("she reads a book", "她读书"),
    ("i write a letter", "我写信"),
    ("he speaks english", "他说英语"),
    ("she speaks chinese", "她说中文"),
    ("i want some tea", "我想要茶"),
    ("i want some coffee", "我想要咖啡"),
    ("can you help me", "你能帮我吗"),
    ("please help me", "请帮我"),
    ("open the door", "开门"),
    ("close the window", "关窗"),
    ("turn on the light", "开灯"),
    ("turn off the light", "关灯"),
    ("it is sunny today", "今天天气晴朗"),
    ("it is raining", "正在下雨"),
    ("i feel happy", "我感到开心"),
    ("i feel sad", "我感到难过"),
    ("he is very tall", "他很高"),
    ("she is very beautiful", "她很漂亮"),
    ("the food is delicious", "食物很美味"),
    ("the movie is interesting", "电影很有趣"),
    ("i study chinese", "我学中文"),
    ("i study english", "我学英语"),
    ("he plays football", "他踢足球"),
    ("she plays the piano", "她弹钢琴"),
    ("we watch a movie", "我们看电影"),
    ("they play games", "他们玩游戏"),
    ("i have a pen", "我有一支笔"),
    ("he has a car", "他有一辆车"),
    ("she has a red bag", "她有一个红包"),
    ("there is a bird", "有一只鸟"),
    ("there are two cats", "有两只猫"),
    ("i need your help", "我需要你的帮助"),
    ("do you like apples", "你喜欢苹果吗"),
    ("yes i like apples", "是的我喜欢苹果"),
    ("no i do not like tea", "不我不喜欢茶"),
    ("what time is it", "现在几点"),
    ("it is eight o clock", "现在八点"),
    ("let us go home", "我们回家吧"),
    ("wait for me", "等我"),
    ("come with me", "跟我来"),
    ("sit down please", "请坐"),
    ("stand up please", "请站起来"),
    ("i am learning machine translation", "我在学习机器翻译"),
    ("neural networks are powerful", "神经网络很强大"),
    ("attention is all you need", "注意力就是你所需要的一切"),
    ("deep learning changes the world", "深度学习改变世界"),
    ("artificial intelligence is useful", "人工智能很有用"),
    ("i train a neural network", "我训练一个神经网络"),
    ("the model generates nice sentences", "模型生成漂亮的句子"),
]


# ---------------------------------------------------------------------------
# 分词
# ---------------------------------------------------------------------------

def tokenize_en(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def tokenize_zh(text: str) -> list[str]:
    """按字分词，标点保留。"""
    text = text.strip()
    return list(text) if text else []


# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------

class Vocab:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    @property
    def pad_idx(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_idx(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_idx(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_idx(self) -> int:
        return self.token_to_id[UNK]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str], add_bos_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_idx) for t in tokens]
        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens = []
        special = {self.pad_idx, self.bos_idx, self.eos_idx}
        for i in ids:
            if skip_special and i in special:
                continue
            if i == self.eos_idx and skip_special:
                break
            tokens.append(self.id_to_token.get(int(i), UNK))
        # 中文按字拼接，英文空格拼接：简单启发式——无空格 token 视为中文
        if tokens and all(len(t) == 1 or t in SPECIAL_TOKENS for t in tokens):
            return "".join(tokens)
        return " ".join(tokens)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def build_vocab(token_lists: list[list[str]], min_freq: int = 1) -> Vocab:
    counter: Counter[str] = Counter()
    for toks in token_lists:
        counter.update(toks)
    token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    return Vocab(token_to_id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class ParallelExample:
    src: str
    tgt: str


class TranslationDataset(Dataset):
    def __init__(self, pairs: list[ParallelExample], src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ex = self.pairs[idx]
        src_ids = self.src_vocab.encode(tokenize_en(ex.src), add_bos_eos=True)
        tgt_ids = self.tgt_vocab.encode(tokenize_zh(ex.tgt), add_bos_eos=True)
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": ex.src,
            "tgt_text": ex.tgt,
        }


def collate_fn(batch, pad_idx: int = 0):
    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=pad_idx)
    tgt = pad_sequence([b["tgt"] for b in batch], batch_first=True, padding_value=pad_idx)
    return {
        "src": src,
        "tgt": tgt,
        "src_text": [b["src_text"] for b in batch],
        "tgt_text": [b["tgt_text"] for b in batch],
    }


# ---------------------------------------------------------------------------
# 加载数据
# ---------------------------------------------------------------------------

def load_builtin_pairs() -> list[ParallelExample]:
    return [ParallelExample(s, t) for s, t in BUILTIN_PAIRS]


def load_tsv_pairs(path: Path) -> list[ParallelExample]:
    """TSV 格式：src\\ttgt，一行一对。"""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            pairs.append(ParallelExample(parts[0].strip(), parts[1].strip()))
    return pairs


def prepare_data(
    source: str = "builtin",
    tsv_path: Path | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    返回 train_loader 构建所需的 datasets + vocabs。
    source: 'builtin' | 'tsv'
    """
    if source == "builtin":
        pairs = load_builtin_pairs()
    elif source == "tsv":
        assert tsv_path is not None and tsv_path.exists()
        pairs = load_tsv_pairs(tsv_path)
    else:
        raise ValueError(f"未知数据源: {source}")

    # 划分训练/验证
    g = torch.Generator().manual_seed(seed)
    n = len(pairs)
    n_val = max(1, int(n * val_ratio))
    perm = torch.randperm(n, generator=g).tolist()
    val_pairs = [pairs[i] for i in perm[:n_val]]
    train_pairs = [pairs[i] for i in perm[n_val:]]

    # 仅用训练集建词表
    src_vocab = build_vocab([tokenize_en(p.src) for p in train_pairs])
    tgt_vocab = build_vocab([tokenize_zh(p.tgt) for p in train_pairs])

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(val_pairs, src_vocab, tgt_vocab)
    return train_ds, val_ds, src_vocab, tgt_vocab


def make_dataloader(dataset: TranslationDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, pad_idx=dataset.src_vocab.pad_idx),
        num_workers=0,
    )
