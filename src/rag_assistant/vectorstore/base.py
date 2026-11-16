"""向量库抽象层。

需求要求"先用 Chroma/Faiss，代码中通过抽象接口封装，便于后续切换 Milvus"。
这里把向量库的能力收敛成 6 个方法：写入、检索、按文档删除、计数、
列出文档 ID、清空。切换 Milvus/Qdrant 时只需新增一个实现类。

约定：
- **score 越大越相关**（相似度语义）。Chroma 返回的是距离，适配层负责换算。
- ``where`` 过滤条件使用本项目的元数据键（``doc_id`` / ``doc_type`` / ...），
  各实现自行翻译为自己的过滤语法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class VectorRecord:
    """一条待写入的向量记录。"""

    chunk_id: str
    text: str
    metadata: Mapping[str, str | int | float]
    embedding: Sequence[float]


@dataclass(frozen=True)
class VectorHit:
    """一条检索结果。"""

    chunk_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))


class VectorStore(ABC):
    """向量库接口。"""

    name: str = "base"

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """写入或覆盖记录，返回写入条数。"""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        """按向量检索，返回按相关度降序的命中列表。"""

    @abstractmethod
    def delete_by_doc(self, doc_ids: Sequence[str]) -> int:
        """删除指定文档的全部向量，返回删除条数（无法精确时返回 -1）。"""

    @abstractmethod
    def count(self) -> int:
        """当前记录总数。"""

    @abstractmethod
    def list_doc_ids(self) -> set[str]:
        """当前已入库的 doc_id 集合。"""

    @abstractmethod
    def reset(self) -> None:
        """清空所有记录（重建索引时使用）。"""

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, VectorHit]:
        """按 chunk_id 批量取回原始记录（用于"引用可追到原文"）。

        默认实现返回空字典：不是所有后端都必须支持按 id 回查，
        调用方（``KnowledgeBase.get_chunks``）对缺失的 id 直接跳过。
        """
        return {}

    def close(self) -> None:
        """释放资源（默认无操作）。"""
