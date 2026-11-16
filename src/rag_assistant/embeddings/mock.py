"""确定性哈希 Embedding（mock 模式）。

用途：
1. 无网络 / 无 GPU 环境跑通全链路与全部测试；
2. CI 中验证"检索管道接线"是否正确（不校验语义质量）。

实现：字符 n-gram（n=1..3）→ blake2b 哈希 → 带符号分桶 → L2 归一。
- 用 blake2b 而不是内置 ``hash()``：后者带进程随机盐，跨进程不可复现。
- 带符号哈希（signed hashing）能抵消部分哈希碰撞带来的虚假相似度。
- 同义不同字（"休学" vs "暂停学业"）召回不了，这是刻意的：mock 不假装有语义能力。
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from .base import Embedder

_NGRAM_SIZES = (1, 2, 3)


def _hash_bucket(gram: str, dim: int) -> tuple[int, float]:
    """把一个 n-gram 映射到 (桶下标, ±1)。"""
    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    bucket = value % dim
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return bucket, sign


class HashEmbedder(Embedder):
    """确定性哈希向量，接口与真实模型完全一致。"""

    name = "hash-mock"

    def __init__(self, dim: int = 1024, normalize: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim
        self._normalize = normalize

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        cleaned = (text or "").strip().lower()
        if not cleaned:
            return vector
        for size in _NGRAM_SIZES:
            if len(cleaned) < size:
                continue
            for start in range(len(cleaned) - size + 1):
                gram = cleaned[start : start + size]
                if gram.isspace():
                    continue
                bucket, sign = _hash_bucket(gram, self.dim)
                vector[bucket] += sign
        if self._normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @property
    def is_mock(self) -> bool:
        return True
