"""检索结果模型：把向量/BM25/rerank 三路分数集中在一个对象上。

字段设计原则：**保留每一步的分数**，不只是最终分。这样：
- 调阈值时能看出"是召不回来"还是"召回了但分低"；
- 日志里能复盘某次拒答到底卡在哪一环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RetrievedChunk:
    """一条召回结果（自带溯源字段）。"""

    chunk_id: str
    doc_id: str
    doc_name: str
    doc_type: str
    section_path: str
    text: str
    page: int = 0
    chunk_type: str = "text"
    file_version: str = ""
    source_path: str = ""
    order: int = 0
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        """用于排序与阈值比较的最终分（有 rerank 用 rerank，否则用 RRF）。"""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.rrf_score

    @property
    def page_label(self) -> str:
        return f"第 {self.page} 页" if self.page else "无页码"

    @property
    def location(self) -> str:
        """人类可读的位置描述：章节路径（+页码）。"""
        if self.section_path:
            return self.section_path if not self.page else f"{self.section_path}（{self.page_label}）"
        return self.page_label

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "section_path": self.section_path,
            "page": self.page,
            "chunk_type": self.chunk_type,
            "text": self.text,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "rrf_score": round(self.rrf_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
        }

    @classmethod
    def from_metadata(
        cls,
        chunk_id: str,
        text: str,
        metadata: Mapping[str, Any],
        *,
        vector_score: float | None = None,
        bm25_score: float | None = None,
    ) -> "RetrievedChunk":
        """从向量库/BM25 的 metadata 还原对象（两者字段一致）。"""
        return cls(
            chunk_id=chunk_id,
            doc_id=str(metadata.get("doc_id", "")),
            doc_name=str(metadata.get("doc_name", "")),
            doc_type=str(metadata.get("doc_type", "")),
            section_path=str(metadata.get("section_path", "")),
            text=text,
            page=int(metadata.get("page", 0) or 0),
            chunk_type=str(metadata.get("chunk_type", "text")),
            file_version=str(metadata.get("file_version", "")),
            source_path=str(metadata.get("source_path", "")),
            order=int(metadata.get("order", 0) or 0),
            vector_score=vector_score,
            bm25_score=bm25_score,
        )


@dataclass
class RetrievalResult:
    """一次检索的完整结果。"""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    best_score: float = 0.0
    has_evidence: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def doc_ids(self) -> list[str]:
        return [chunk.doc_id for chunk in self.chunks]

    @property
    def best_vector_score(self) -> float:
        scores = [c.vector_score for c in self.chunks if c.vector_score is not None]
        return max(scores) if scores else 0.0
