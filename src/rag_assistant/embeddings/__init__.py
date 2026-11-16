"""Embedding 工厂。"""

from __future__ import annotations

from ..config import EmbeddingConfig, Settings
from .base import Embedder, resolve_device
from .bge import BgeEmbedder
from .mock import HashEmbedder

__all__ = ["BgeEmbedder", "Embedder", "HashEmbedder", "get_embedder", "resolve_device"]


def get_embedder(settings: Settings) -> Embedder:
    """按配置返回 Embedding 实现。

    判定顺序：
    1. ``embedding.provider == mock`` → 哈希实现
    2. ``RAG_FORCE_MOCK=1`` 已把 provider 改成 mock（由 config 层完成）
    3. 其他 → bge-m3（真实模型，首次调用才加载）
    """
    config: EmbeddingConfig = settings.embedding
    provider = config.provider.strip().lower()
    if provider in {"mock", "hash"}:
        return HashEmbedder(dim=config.dim, normalize=config.normalize)
    if provider in {"bge", "sentence-transformers", "sentence_transformers"}:
        return BgeEmbedder(config)
    raise ValueError(f"未知的 embedding.provider：{config.provider}")
