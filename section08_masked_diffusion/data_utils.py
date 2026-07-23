# -*- coding: utf-8 -*-
"""
Section 08 - 离散 Mask 扩散的语料与词表

复用 Section 06 的英文短句作「语言模型」语料（无需联网）。
特殊符号多一个 <mask>：这就是离散扩散里的「噪声态」。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent

PAD, BOS, EOS, UNK, MASK = "<pad>", "<bos>", "<eos>", "<unk>", "<mask>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK, MASK]


def setup_stdio() -> None:
    """Windows GBK 控制台下避免 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# 取自 Section 06 英→中平行语料的英文侧（教学用短句）
BUILTIN_SENTENCES: list[str] = [
    "i love you",
    "i like cats",
    "i like dogs",
    "he likes music",
    "she likes books",
    "we love china",
    "they like tea",
    "i am a student",
    "he is a teacher",
    "she is a doctor",
    "we are friends",
    "you are kind",
    "this is a book",
    "that is a cat",
    "where is the book",
    "where is the school",
    "what is your name",
    "my name is tom",
    "how are you",
    "i am fine",
    "thank you",
    "you are welcome",
    "good morning",
    "good night",
    "see you tomorrow",
    "i go to school",
    "he goes to work",
    "she goes home",
    "we go to the park",
    "i eat an apple",
    "he drinks water",
    "she reads a book",
    "i write a letter",
    "he speaks english",
    "she speaks chinese",
    "i want some tea",
    "i want some coffee",
    "can you help me",
    "please help me",
    "open the door",
    "close the window",
    "turn on the light",
    "turn off the light",
    "it is sunny today",
    "it is raining",
    "i feel happy",
    "i feel sad",
    "he is very tall",
    "she is very beautiful",
    "the food is delicious",
    "the movie is interesting",
    "i study chinese",
    "i study english",
    "he plays football",
    "she plays the piano",
    "we watch a movie",
    "they play games",
    "i have a pen",
    "he has a car",
    "she has a red bag",
    "there is a bird",
    "there are two cats",
    "i need your help",
    "do you like apples",
    "yes i like apples",
    "no i do not like tea",
    "what time is it",
    "it is eight o clock",
    "let us go home",
    "wait for me",
    "come with me",
    "sit down please",
    "stand up please",
    "i am learning machine translation",
    "neural networks are powerful",
    "attention is all you need",
    "deep learning changes the world",
    "artificial intelligence is useful",
    "i train a neural network",
    "the model generates nice sentences",
]


def tokenize(text: str) -> list[str]:
    return text.lower().strip().split()


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_idx: int
    bos_idx: int
    eos_idx: int
    unk_idx: int
    mask_idx: int

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str], add_bos_eos: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_idx) for t in tokens]
        if add_bos_eos:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        toks = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            toks.append(tok)
        return " ".join(toks)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "token_to_id": self.token_to_id,
                    "pad_idx": self.pad_idx,
                    "bos_idx": self.bos_idx,
                    "eos_idx": self.eos_idx,
                    "unk_idx": self.unk_idx,
                    "mask_idx": self.mask_idx,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        obj = json.loads(path.read_text(encoding="utf-8"))
        t2i = obj["token_to_id"]
        i2t = {int(v): k for k, v in t2i.items()}
        return cls(
            token_to_id=t2i,
            id_to_token=i2t,
            pad_idx=obj["pad_idx"],
            bos_idx=obj["bos_idx"],
            eos_idx=obj["eos_idx"],
            unk_idx=obj["unk_idx"],
            mask_idx=obj["mask_idx"],
        )


def build_vocab(sentences: list[str] | None = None) -> Vocab:
    sentences = sentences or BUILTIN_SENTENCES
    counter: Counter[str] = Counter()
    for s in sentences:
        counter.update(tokenize(s))

    token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok, _ in counter.most_common():
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    id_to_token = {i: t for t, i in token_to_id.items()}
    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_idx=token_to_id[PAD],
        bos_idx=token_to_id[BOS],
        eos_idx=token_to_id[EOS],
        unk_idx=token_to_id[UNK],
        mask_idx=token_to_id[MASK],
    )


class TextDataset(Dataset):
    def __init__(self, sentences: list[str], vocab: Vocab):
        self.vocab = vocab
        self.ids = [vocab.encode(tokenize(s), add_bos_eos=True) for s in sentences]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.ids[idx], dtype=torch.long)


def collate_batch(batch: list[torch.Tensor], pad_idx: int) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True, padding_value=pad_idx)


def make_dataloader(
    vocab: Vocab,
    sentences: list[str] | None = None,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    sentences = sentences or BUILTIN_SENTENCES
    ds = TextDataset(sentences, vocab)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_batch(b, vocab.pad_idx),
    )


def ids_to_readable(ids: list[int] | torch.Tensor, vocab: Vocab) -> str:
    """把 id 序列打印成可读字符串，显式保留 <mask> / <pad> 等。"""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return " ".join(vocab.id_to_token.get(i, UNK) for i in ids)
