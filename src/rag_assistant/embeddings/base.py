"""Embedding 抽象层。

为什么要抽象：需求要求"默认 bge-m3，可通过配置切换"，同时要能离线跑测试。
抽象层提供三种实现：
- :class:`~rag_assistant.embeddings.bge.BgeEmbedder`：真实模型（bge-m3）
- :class:`~rag_assistant.embeddings.mock.HashEmbedder`：确定性哈希向量（无网络/CI）
- 后续接其他模型只需再加一个子类，检索链路不用改。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


def resolve_device(preference: str) -> str:
    """把 ``auto`` 解析成具体设备：有 CUDA 用 GPU，否则 CPU。

    ``torch`` 未安装时（mock 模式）直接返回 cpu，不抛错。
    """
    pref = (preference or "auto").strip().lower()
    if pref in {"cpu", "cuda", "mps"}:
        if pref == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Embedder(ABC):
    """向量化接口。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码文档（入库用）。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """编码单条查询。

        单独开一个方法而不是复用 ``embed_documents``：某些模型（如 bge-v1.5 系列）
        对查询需要加指令前缀，bge-m3 则不需要 —— 差异集中在这里处理。
        """

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码多条查询（multi-query 场景），默认逐条编码。"""
        return [self.embed_query(text) for text in texts]

    @property
    def is_mock(self) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<{type(self).__name__} name={self.name} dim={self.dim}>"
