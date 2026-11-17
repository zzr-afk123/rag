"""混合检索编排：向量 + BM25 → RRF → Rerank → 阈值 → Top-N。

链路（每一步都能在 ``diagnostics`` 里看到数量与耗时）::

    query (+ multi-query 改写)
        ├── 向量检索 (candidate_k) ──┐
        └── BM25   (candidate_k) ────┴── RRF 融合 (k=60, 权重可配)
                                          └── Rerank 前 top_k_in 条
                                                └── 阈值判定 → Top-N

阈值口径（方案假设 #4 的落地）：
- 有 reranker 时用 **rerank 的 sigmoid 分数**（0–1，可跨查询比较）；
- 关闭 reranker 时退化为 **top-1 的向量余弦相似度**（也在 0–1）。
  这种情况下 RRF 分数不参与阈值，因为它是"名次倒数之和"，
  数值大小只跟名次有关，跟"像不像"无关，绝不能当相似度用。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..config import Settings
from ..embeddings.base import Embedder
from ..logging_setup import get_logger
from ..vectorstore.base import VectorStore
from .bm25 import BM25Index
from .fusion import rrf_fuse
from .models import RetrievalResult, RetrievedChunk
from .reranker import Reranker

logger = get_logger("retrieval.retriever")

#: 疑问/虚化词：不参与"词面覆盖"判断。
#: 这些词几乎每句话都有、却极少出现在制度正文里，计入分母会把正常问题判成无关。
_QUESTION_WORDS = frozenset(
    {
        "多久", "多长时间", "多少", "多长", "什么", "哪些", "哪个", "哪里", "在哪",
        "几点", "几号", "怎么", "怎样", "如何", "为什么", "为何", "是否", "可以",
        "能否", "需要", "要不要", "有没有", "是谁", "谁", "请问", "告诉", "一下",
        "相关", "具体", "一般", "规定", "政策", "办法",
    }
)


class HybridRetriever:
    """混合检索器。"""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker | None = None,
        lock: object | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._bm25 = bm25
        self._reranker = reranker
        #: 可选的索引锁（阶段 E）。约定接口：``read()`` 上下文管理器。
        self._lock = lock

    # ---- 单路召回 ----
    def _vector_recall(self, query: str) -> tuple[list[str], dict[str, RetrievedChunk]]:
        embedding = self._embedder.embed_query(query)
        hits = self._vector_store.query(embedding, self._settings.retrieval.candidate_k)
        ids: list[str] = []
        payload: dict[str, RetrievedChunk] = {}
        for hit in hits:
            ids.append(hit.chunk_id)
            payload[hit.chunk_id] = RetrievedChunk.from_metadata(
                hit.chunk_id, hit.text, hit.metadata, vector_score=hit.score
            )
        return ids, payload

    def _bm25_recall(self, query: str) -> tuple[list[str], dict[str, RetrievedChunk]]:
        hits = self._bm25.search(query, self._settings.retrieval.candidate_k)
        ids: list[str] = []
        payload: dict[str, RetrievedChunk] = {}
        for hit in hits:
            ids.append(hit.chunk_id)
            payload[hit.chunk_id] = RetrievedChunk.from_metadata(
                hit.chunk_id, hit.text, hit.metadata, bm25_score=hit.score
            )
        return ids, payload

    # ---- 主流程 ----
    def retrieve(
        self,
        query: str,
        *,
        top_n: int | None = None,
        extra_queries: Sequence[str] = (),
        threshold: float | None = None,
    ) -> RetrievalResult:
        """执行混合检索。

        查询期间持**读锁**：同步任务正在提交时必须等它提交完，避免读到
        "旧 chunk 已删、新 chunk 未写"的中间态。等不到锁会抛
        :class:`~rag_assistant.exceptions.IndexBusyError`（API 映射为 409）。

        Args:
            query: 主查询（已由上层完成指代消解）。
            top_n: 返回条数，缺省用配置值。
            extra_queries: multi-query 生成的等价改写，分别召回后一起融合。
            threshold: 临时覆盖 ``retrieval.score_threshold``（标定阈值时使用）。

        Returns:
            :class:`RetrievalResult`；``has_evidence`` 为 False 表示证据不足，
            调用方应走拒答流程（但仍可用 ``chunks`` 给"最相近文档线索"）。
        """
        if self._lock is None:
            return self._retrieve(query, top_n=top_n, extra_queries=extra_queries, threshold=threshold)
        with self._lock.read():  # type: ignore[attr-defined]
            return self._retrieve(query, top_n=top_n, extra_queries=extra_queries, threshold=threshold)

    def _retrieve(
        self,
        query: str,
        *,
        top_n: int | None = None,
        extra_queries: Sequence[str] = (),
        threshold: float | None = None,
    ) -> RetrievalResult:
        """检索主体（不加锁）。"""
        started = time.perf_counter()
        limit = top_n or self._settings.retrieval.top_n
        queries = [query] + [q.strip() for q in extra_queries if q and q.strip() and q.strip() != query]

        vector_rankings: list[list[str]] = []
        bm25_rankings: list[list[str]] = []
        payload: dict[str, RetrievedChunk] = {}
        vector_hits_total = 0
        bm25_hits_total = 0

        for single in queries:
            vec_ids, vec_payload = self._vector_recall(single)
            vector_hits_total += len(vec_ids)
            if vec_ids:
                vector_rankings.append(vec_ids)
                payload.update({cid: chunk for cid, chunk in vec_payload.items() if cid not in payload})

            kw_ids, kw_payload = self._bm25_recall(single)
            bm25_hits_total += len(kw_ids)
            if kw_ids:
                bm25_rankings.append(kw_ids)
                for cid, chunk in kw_payload.items():
                    existing = payload.get(cid)
                    if existing is None:
                        payload[cid] = chunk
                    elif existing.bm25_score is None:
                        # 同一 chunk 可能先被向量召回：把 BM25 分补上，保留已有向量分
                        payload[cid] = RetrievedChunk(
                            **{**existing.__dict__, "bm25_score": chunk.bm25_score}
                        )

        if not payload:
            logger.info("检索无命中：query=%s", query)
            return RetrievalResult(
                query=query,
                has_evidence=False,
                diagnostics={
                    "queries": queries,
                    "vector_hits": 0,
                    "bm25_hits": 0,
                    "fused": 0,
                    "gate": "none",
                    "threshold": (
                        self._settings.retrieval.score_threshold if threshold is None else float(threshold)
                    ),
                    "best_score": 0.0,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    "reason": "no_candidates",
                },
            )

        weights = [self._settings.retrieval.weight_vector] * len(vector_rankings) + [
            self._settings.retrieval.weight_bm25
        ] * len(bm25_rankings)
        fused = rrf_fuse(
            [*vector_rankings, *bm25_rankings],
            weights=weights,
            k=self._settings.retrieval.rrf_k,
        )

        fused_chunks: list[RetrievedChunk] = []
        for chunk_id, score in fused:
            chunk = payload.get(chunk_id)
            if chunk is None:
                continue
            fused_chunks.append(RetrievedChunk(**{**chunk.__dict__, "rrf_score": score}))

        rerank_window = self._settings.reranker.top_k_in if self._reranker else 0
        if self._reranker and rerank_window > 0:
            window = fused_chunks[:rerank_window]
            scores = self._reranker.score(query, [chunk.text for chunk in window])
            fused_chunks = [
                RetrievedChunk(**{**chunk.__dict__, "rerank_score": score})
                for chunk, score in zip(window, scores)
            ]
            gate = "rerank"
        else:
            gate = "vector_cosine"

        fused_chunks.sort(key=lambda item: item.final_score, reverse=True)

        threshold_value = (
            self._settings.retrieval.score_threshold if threshold is None else float(threshold)
        )
        if gate == "rerank":
            best = fused_chunks[0].final_score if fused_chunks else 0.0
        else:
            best = max((chunk.vector_score or 0.0) for chunk in fused_chunks) if fused_chunks else 0.0
        # ---- 词面覆盖闸门（解决"问天气却答休学"）----
        # 分数阈值挡不住无关问题：向量/BM25 永远会返回 top-k，重排也会给一个分数。
        # 这里额外要求"问题的内容词确实出现在召回片段里"，否则判 off_topic。
        coverage, content_tokens = self._coverage(query, fused_chunks[: max(3, self._settings.retrieval.top_n)])
        min_coverage = self._settings.retrieval.min_lexical_coverage
        relax_score = self._settings.retrieval.coverage_relax_score
        # 精排已经判定"这段证据与问题相关"时，词面重叠不再作为必要条件：
        # 口语化提问的内容词往往一个都不在公文原文里，见 config.RetrievalConfig 的注释。
        relaxed_by_rerank = bool(
            gate == "rerank"
            and relax_score > 0
            and self._reranker is not None
            and not self._reranker.is_mock
            and best >= relax_score
        )
        coverage_floor = 0.0 if relaxed_by_rerank else min_coverage
        off_topic = bool(content_tokens) and coverage_floor > 0 and coverage < coverage_floor

        has_evidence = bool(fused_chunks) and best >= threshold_value and not off_topic

        is_mock_chain = self._embedder.is_mock and (self._reranker is None or self._reranker.is_mock)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        result = RetrievalResult(
            query=query,
            chunks=fused_chunks[:limit],
            best_score=round(best, 6),
            has_evidence=has_evidence,
            diagnostics={
                "queries": queries,
                "vector_hits": vector_hits_total,
                "bm25_hits": bm25_hits_total,
                "fused": len(fused),
                "reranked": len(fused_chunks) if gate == "rerank" else 0,
                "gate": gate,
                "threshold": threshold_value,
                "threshold_source": "override" if threshold is not None else "config",
                "best_score": round(best, 6),
                "reranker": self._reranker.name if self._reranker else None,
                "lexical_coverage": round(coverage, 3),
                "coverage_floor": coverage_floor,
                "coverage_relaxed": relaxed_by_rerank,
                "content_tokens": content_tokens,
                "off_topic": off_topic,
                "latency_ms": latency_ms,
                "reason": "off_topic" if off_topic else ("ok" if has_evidence else "below_threshold"),
                "note": (
                    "mock 链路：分数口径与真实 bge 模型不可比，阈值仅用于验证链路连通性"
                    if is_mock_chain
                    else None
                ),
            },
        )
        logger.info(
            "检索完成 query=%r 候选=%d(向量 %d/BM25 %d) 融合=%d 最优分=%.4f 门槛=%.2f 有依据=%s 耗时=%sms",
            query,
            len(payload),
            vector_hits_total,
            bm25_hits_total,
            len(fused),
            best,
            threshold_value,
            has_evidence,
            latency_ms,
        )
        return result

    def _coverage(self, query: str, chunks: Sequence[RetrievedChunk]) -> tuple[float, list[str]]:
        """问题内容词在召回片段中的覆盖率（0–1）。

        只统计"内容词"：疑问词与虚化词（多久/怎么/可以/规定…）不计入分母，
        否则"休学最长可以多久"这种正常问题会因为"可以/多久"没出现而被误拒。
        """
        tokenizer = getattr(self._bm25, "tokenizer", None)
        if tokenizer is None or not chunks:
            return 0.0, []
        tokens = [token for token in tokenizer.tokenize(query) if token not in _QUESTION_WORDS]
        unique = list(dict.fromkeys(tokens))
        if not unique:
            return 0.0, []
        found: set[str] = set()
        for chunk in chunks:
            found |= set(unique) & set(tokenizer.tokenize(chunk.text))
            if len(found) == len(unique):
                break
        return len(found) / len(unique), unique

    # ---- 供上层（阶段 C/D）使用 ----
    def close(self) -> None:
        """释放底层资源。"""
        self._vector_store.close()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "chunks": self._vector_store.count(),
            "bm25_chunks": self._bm25.count,
            "embedder": self._embedder.name,
            "reranker": self._reranker.name if self._reranker else None,
        }
