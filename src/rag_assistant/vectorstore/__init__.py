"""向量库工厂。"""

from __future__ import annotations

from ..config import Settings
from .base import VectorHit, VectorRecord, VectorStore
from .chroma_store import ChromaVectorStore
from .memory_store import MemoryVectorStore

__all__ = [
    "ChromaVectorStore",
    "MemoryVectorStore",
    "VectorHit",
    "VectorRecord",
    "VectorStore",
    "get_vector_store",
]


def get_vector_store(settings: Settings, *, recreate: bool = False) -> VectorStore:
    """按配置返回向量库实现。

    ``backend`` 支持 ``chroma`` / ``memory``；``memory`` 仅用于测试与快速演示。
    """
    backend = settings.vectorstore.backend.strip().lower()
    if backend == "memory":
        return MemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(
            settings.chroma_dir,
            settings.collection_name,
            recreate=recreate,
        )
    raise ValueError(f"未知的 vectorstore.backend：{settings.vectorstore.backend}")
