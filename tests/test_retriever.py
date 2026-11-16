"""混合检索链路测试：召回 → RRF → Rerank → 阈值判定。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings.mock import HashEmbedder
from rag_assistant.retrieval.bm25 import BM25Index, BM25Record, ChineseTokenizer
from rag_assistant.retrieval.reranker import MockReranker
from rag_assistant.retrieval.retriever import HybridRetriever
from rag_assistant.vectorstore.base import VectorRecord
from rag_assistant.vectorstore.memory_store import MemoryVectorStore

CORPUS: list[tuple[str, str, str]] = [
    (
        "d1#1",
        "d1",
        "学生因病需停课治疗的应当办理休学。休学期一般以一年为期，累计不得超过两年。",
    ),
    (
        "d1#2",
        "d1",
        "休学期满的学生应当在开学前一个月内提出复学申请，并提交康复证明。",
    ),
    (
        "d2#1",
        "d2",
        "在读证明可线上办理，1 个工作日内审核；线下携带学生证到教务处学籍科立等可取。",
    ),
    (
        "d3#1",
        "d3",
        "科研经费差旅费报销须在出差结束后 15 个工作日内办理，住宿费按标准限额内据实报销。",
    ),
]


def build_retriever(
    settings: Settings,
    repo_root: Path,
    *,
    with_reranker: bool = True,
    threshold: float = 0.5,
) -> HybridRetriever:
    """构造一条最小可用的检索链路（内存向量库 + 真实 BM25 + mock 重排）。"""
    embedder = HashEmbedder(dim=128)
    store = MemoryVectorStore()
    tokenizer = ChineseTokenizer(
        userdict_path=repo_root / "config" / "jieba_userdict.txt",
        stopwords_path=repo_root / "config" / "stopwords_zh.txt",
    )
    bm25 = BM25Index(tokenizer)

    texts = [text for _, _, text in CORPUS]
    vectors = embedder.embed_documents(texts)
    store.upsert(
        [
            VectorRecord(
                chunk_id=chunk_id,
                text=text,
                metadata={"doc_id": doc_id, "doc_name": f"{doc_id}.md", "section_path": "章节", "page": 0},
                embedding=vector,
            )
            for (chunk_id, doc_id, text), vector in zip(CORPUS, vectors)
        ]
    )
    bm25.add([BM25Record(chunk_id=cid, text=text, metadata={"doc_id": doc}) for cid, doc, text in CORPUS])

    # 用 dataclasses.replace 派生一份配置（Settings 是 frozen 的，不就地改）
    tuned = replace(settings, retrieval=replace(settings.retrieval, score_threshold=threshold))
    return HybridRetriever(tuned, embedder, store, bm25, MockReranker() if with_reranker else None)


def test_retrieve_returns_top_n(settings: Settings, repo_root: Path) -> None:
    retriever = build_retriever(settings, repo_root)
    result = retriever.retrieve("休学最长多久", top_n=2)

    assert len(result.chunks) == 2
    assert result.chunks[0].doc_id == "d1"
    assert result.diagnostics["vector_hits"] > 0
    assert result.diagnostics["bm25_hits"] > 0
    assert result.diagnostics["gate"] == "rerank"


def test_rerank_scores_present_and_ordered(settings: Settings, repo_root: Path) -> None:
    retriever = build_retriever(settings, repo_root)
    result = retriever.retrieve("休学最长多久", top_n=4)

    assert all(chunk.rerank_score is not None for chunk in result.chunks)
    scores = [chunk.rerank_score for chunk in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_threshold_gate_refuses(settings: Settings, repo_root: Path) -> None:
    """阈值高于最优分 → 判定证据不足，但仍返回线索。"""
    retriever = build_retriever(settings, repo_root, threshold=0.99)
    result = retriever.retrieve("休学最长多久")

    assert result.has_evidence is False
    assert result.diagnostics["reason"] == "below_threshold"
    assert result.chunks  # 线索仍要给，供"最相近文档"提示
    assert result.best_score < 0.99


def test_threshold_gate_passes(settings: Settings, repo_root: Path) -> None:
    retriever = build_retriever(settings, repo_root, threshold=0.0)
    result = retriever.retrieve("休学最长多久")
    assert result.has_evidence is True
    assert result.diagnostics["reason"] == "ok"


def test_threshold_override_argument(settings: Settings, repo_root: Path) -> None:
    """retrieve(threshold=...) 能临时覆盖配置（阶段 F 标定用）。"""
    retriever = build_retriever(settings, repo_root, threshold=0.0)
    result = retriever.retrieve("休学最长多久", threshold=0.99)
    assert result.has_evidence is False
    assert result.diagnostics["threshold"] == 0.99
    assert result.diagnostics["threshold_source"] == "override"


def test_without_reranker_uses_vector_gate(settings: Settings, repo_root: Path) -> None:
    """关闭 rerank 时退化为向量相似度口径，且不产生 rerank_score。"""
    retriever = build_retriever(settings, repo_root, with_reranker=False, threshold=0.1)
    result = retriever.retrieve("休学最长多久")

    assert result.diagnostics["gate"] == "vector_cosine"
    assert all(chunk.rerank_score is None for chunk in result.chunks)
    expected = max(chunk.vector_score or 0 for chunk in result.chunks)
    assert result.best_score == pytest.approx(expected, abs=1e-4)


def test_multi_query_increases_candidates(settings: Settings, repo_root: Path) -> None:
    """额外改写查询能补召回（原查询漏掉时靠改写兜住）。"""
    retriever = build_retriever(settings, repo_root, threshold=0.0)
    single = retriever.retrieve("休学", top_n=4)
    multi = retriever.retrieve("休学", top_n=4, extra_queries=["复学申请材料", "科研经费报销"])

    assert len(multi.diagnostics["queries"]) == 3
    assert multi.diagnostics["fused"] >= single.diagnostics["fused"]


def test_empty_index(settings: Settings, repo_root: Path) -> None:
    """空索引：不崩、明确标记 no_candidates。"""
    retriever = HybridRetriever(
        settings,
        HashEmbedder(dim=32),
        MemoryVectorStore(),
        BM25Index(
            ChineseTokenizer(
                userdict_path=repo_root / "config" / "jieba_userdict.txt",
                stopwords_path=repo_root / "config" / "stopwords_zh.txt",
            )
        ),
        MockReranker(),
    )
    result = retriever.retrieve("休学多久")
    assert result.chunks == []
    assert result.has_evidence is False
    assert result.diagnostics["reason"] == "no_candidates"


def test_scores_merged_across_sources(settings: Settings, repo_root: Path) -> None:
    """同一 chunk 被两路召回时，两路分数都要保留（便于排查）。"""
    retriever = build_retriever(settings, repo_root, threshold=0.0)
    result = retriever.retrieve("休学 复学 申请", top_n=4)
    both = [c for c in result.chunks if c.vector_score is not None and c.bm25_score is not None]
    assert both, "应当至少有一条同时被向量与 BM25 召回"


def test_diagnostics_contains_latency(settings: Settings, repo_root: Path) -> None:
    retriever = build_retriever(settings, repo_root)
    result = retriever.retrieve("休学")
    assert result.diagnostics["latency_ms"] >= 0
    assert result.diagnostics["note"], "mock 链路应给出分数不可比的提示"

# --------------------------------------------------------------------------
# 词面覆盖闸门：挡住"问天气却答休学"
# --------------------------------------------------------------------------
def test_off_topic_query_is_refused(settings: Settings, repo_root: Path) -> None:
    """分数阈值设 0 时，无关问题仍必须被词面闸门挡住。"""
    retriever = build_retriever(settings, repo_root, threshold=0.0)
    result = retriever.retrieve("郑州今天天气怎么样")
    assert result.has_evidence is False
    assert result.diagnostics["reason"] == "off_topic"
    assert result.diagnostics["content_tokens"], "郑州/天气应被识别为内容词"
    assert result.diagnostics["lexical_coverage"] < 0.3


def test_on_topic_query_survives_lexical_gate(settings: Settings, repo_root: Path) -> None:
    """疑问词（可以/多久）不该把正常问题拉成无关。"""
    retriever = build_retriever(settings, repo_root, threshold=0.0)
    result = retriever.retrieve("休学最长可以多久")
    assert result.diagnostics["off_topic"] is False
    assert result.has_evidence is True


def test_lexical_gate_can_be_disabled(settings: Settings, repo_root: Path) -> None:
    from dataclasses import replace

    relaxed = replace(settings, retrieval=replace(settings.retrieval, min_lexical_coverage=0.0))
    retriever = build_retriever(relaxed, repo_root, threshold=0.0)
    result = retriever.retrieve("郑州今天天气怎么样")
    assert result.diagnostics["off_topic"] is False
