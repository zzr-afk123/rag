"""向量库测试：内存实现（全量断言）+ Chroma 实现（集成）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.vectorstore.base import VectorRecord
from rag_assistant.vectorstore.memory_store import MemoryVectorStore, match_metadata


def _record(chunk_id: str, doc_id: str, embedding: list[float], text: str = "") -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        text=text or chunk_id,
        metadata={"doc_id": doc_id, "page": 0, "doc_type": "policy"},
        embedding=embedding,
    )


# --------------------------------------------------------------------------
# 内存实现
# --------------------------------------------------------------------------
def test_memory_upsert_and_query() -> None:
    """按余弦相似度降序返回，且覆盖同 ID。"""
    store = MemoryVectorStore()
    store.upsert(
        [
            _record("a#1", "a", [1.0, 0.0]),
            _record("b#1", "b", [0.0, 1.0]),
            _record("c#1", "c", [0.7, 0.7]),
        ]
    )
    hits = store.query([1.0, 0.0], top_k=3)
    assert [hit.chunk_id for hit in hits] == ["a#1", "c#1", "b#1"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[2].score == pytest.approx(0.0, abs=1e-9)

    store.upsert([_record("a#1", "a", [0.0, 1.0], text="覆盖后的文本")])
    assert store.count() == 3
    assert store.query([0.0, 1.0], top_k=1)[0].text == "覆盖后的文本"


def test_memory_where_filter() -> None:
    """元数据过滤：等值 + $in。"""
    store = MemoryVectorStore()
    store.upsert([_record("a#1", "a", [1.0, 0.0]), _record("b#1", "b", [1.0, 0.0])])

    hits = store.query([1.0, 0.0], top_k=5, where={"doc_id": "a"})
    assert [hit.chunk_id for hit in hits] == ["a#1"]

    hits = store.query([1.0, 0.0], top_k=5, where={"doc_id": {"$in": ["b"]}})
    assert [hit.chunk_id for hit in hits] == ["b#1"]


def test_match_metadata_operators() -> None:
    meta = {"doc_id": "a", "page": 3}
    assert match_metadata(meta, None) is True
    assert match_metadata(meta, {"doc_id": "a"}) is True
    assert match_metadata(meta, {"doc_id": "b"}) is False
    assert match_metadata(meta, {"doc_id": {"$in": ["a", "b"]}}) is True
    assert match_metadata(meta, {"page": {"$ne": 3}}) is False


def test_memory_delete_and_reset() -> None:
    store = MemoryVectorStore()
    store.upsert([_record("a#1", "a", [1.0]), _record("a#2", "a", [1.0]), _record("b#1", "b", [1.0])])
    assert store.delete_by_doc(["a"]) == 2
    assert store.count() == 1
    assert store.list_doc_ids() == {"b"}
    store.reset()
    assert store.count() == 0
    assert store.query([1.0], top_k=3) == []


def test_memory_zero_vector_is_safe() -> None:
    """零向量不能触发除零。"""
    store = MemoryVectorStore()
    store.upsert([_record("a#1", "a", [0.0, 0.0])])
    hits = store.query([1.0, 0.0], top_k=1)
    assert hits[0].score == 0.0


def test_memory_top_k_limit() -> None:
    store = MemoryVectorStore()
    store.upsert([_record(f"a#{i}", "a", [float(i), 1.0]) for i in range(1, 6)])
    assert len(store.query([1.0, 1.0], top_k=2)) == 2
    assert store.query([1.0, 1.0], top_k=0) == []


# --------------------------------------------------------------------------
# Chroma 实现（未安装则跳过）
# --------------------------------------------------------------------------
chromadb = pytest.importorskip("chromadb", reason="未安装 chromadb，跳过向量库集成测试")

from rag_assistant.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402


@pytest.fixture()
def chroma_store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(tmp_path / "chroma", "test__dim2", recreate=True)


def test_chroma_roundtrip(chroma_store: ChromaVectorStore) -> None:
    """写入 → 检索（score 越大越相关）→ 删除 → 清空。"""
    written = chroma_store.upsert(
        [
            _record("a#1", "a", [1.0, 0.0], text="在读证明"),
            _record("b#1", "b", [0.0, 1.0], text="休学流程"),
        ]
    )
    assert written == 2
    assert chroma_store.count() == 2

    hits = chroma_store.query([1.0, 0.0], top_k=2)
    assert hits[0].chunk_id == "a#1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert hits[1].score < hits[0].score
    assert hits[0].text == "在读证明"

    assert chroma_store.delete_by_doc(["a"]) == 1
    assert chroma_store.count() == 1
    assert chroma_store.list_doc_ids() == {"b"}

    chroma_store.reset()
    assert chroma_store.count() == 0


def test_chroma_persistence(tmp_path: Path) -> None:
    """重建实例后数据仍在（本地持久化的关键约束）。"""
    path = tmp_path / "chroma"
    store = ChromaVectorStore(path, "persist__test", recreate=True)
    store.upsert([_record("a#1", "a", [1.0, 0.0])])
    store.close()

    reopened = ChromaVectorStore(path, "persist__test")
    assert reopened.count() == 1
    assert reopened.query([1.0, 0.0], top_k=1)[0].chunk_id == "a#1"


def test_chroma_upsert_idempotent(chroma_store: ChromaVectorStore) -> None:
    """同一 chunk_id 重复写入是覆盖而不是追加（重跑同步不会膨胀）。"""
    record = _record("a#1", "a", [1.0, 0.0])
    chroma_store.upsert([record])
    chroma_store.upsert([record])
    assert chroma_store.count() == 1


def test_chroma_empty_query(chroma_store: ChromaVectorStore) -> None:
    """空库查询返回空列表，不抛异常。"""
    assert chroma_store.query([1.0, 0.0], top_k=5) == []
    assert chroma_store.list_doc_ids() == set()
    assert chroma_store.delete_by_doc([]) == 0
