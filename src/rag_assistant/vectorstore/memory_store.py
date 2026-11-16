"""内存向量库（测试与 mock 模式使用）。

不做什么：不做 ANN 索引，纯暴力余弦相似度。数据量小（几千 chunk）时完全够用，
好处是零依赖、行为可预测、便于单元测试断言具体分数顺序。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .base import VectorHit, VectorRecord, VectorStore


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度；任一向量为零向量时返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def match_metadata(metadata: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """支持等值过滤与 ``{"$in": [...]}`` / ``{"$ne": v}``。"""
    if not where:
        return True
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, Mapping):
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            continue
        if actual != expected:
            return False
    return True


class MemoryVectorStore(VectorStore):
    """进程内字典实现。"""

    name = "memory"

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        for record in records:
            self._records[record.chunk_id] = record
        return len(records)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        scored: list[VectorHit] = []
        for record in self._records.values():
            if not match_metadata(record.metadata, where):
                continue
            score = _cosine(embedding, record.embedding)
            scored.append(
                VectorHit(
                    chunk_id=record.chunk_id,
                    text=record.text,
                    metadata=dict(record.metadata),
                    score=score,
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: max(0, top_k)]

    def delete_by_doc(self, doc_ids: Sequence[str]) -> int:
        targets = set(doc_ids)
        to_remove = [cid for cid, rec in self._records.items() if rec.metadata.get("doc_id") in targets]
        for chunk_id in to_remove:
            del self._records[chunk_id]
        return len(to_remove)

    def count(self) -> int:
        return len(self._records)

    def list_doc_ids(self) -> set[str]:
        return {str(rec.metadata.get("doc_id", "")) for rec in self._records.values() if rec.metadata.get("doc_id")}

    def reset(self) -> None:
        self._records.clear()

    def get(self, chunk_id: str) -> VectorRecord | None:
        """测试辅助：按 ID 取记录。"""
        return self._records.get(chunk_id)

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, VectorHit]:
        """按 id 批量取回记录（引用回查用）。"""
        hits: dict[str, VectorHit] = {}
        for chunk_id in chunk_ids:
            record = self._records.get(str(chunk_id))
            if record is None:
                continue
            hits[str(chunk_id)] = VectorHit(
                chunk_id=record.chunk_id,
                text=record.text,
                metadata=record.metadata,
                score=0.0,
            )
        return hits
