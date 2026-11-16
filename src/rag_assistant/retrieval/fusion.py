"""RRF（Reciprocal Rank Fusion）融合。

公式：``score(d) = Σ_i w_i / (k + rank_i(d))``，rank 从 1 开始。

为什么用 RRF 而不是加权分数相加：向量余弦分与 BM25 分**量纲不同**
（前者 0–1，后者可能 0–30+），直接相加会让 BM25 单方面主导。RRF 只用名次，
天然免疫量纲问题，且不需要对两路分数做归一化。

为什么不用 LangChain 的 ``EnsembleRetriever``：它强制等权、且绑定自己的
retriever 协议，无法在这里做"多查询 × 双路"的通用融合，也不好单测。
"""

from __future__ import annotations

from typing import Iterable, Sequence


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """融合多路有序 ID 列表。

    Args:
        rankings: 每路召回的 ID 列表（已按相关度降序，内部会去重）。
        weights: 与 ``rankings`` 等长的权重；缺省全为 1.0。
        k: RRF 平滑常数（默认 60，来自原论文经验值），越大越"平滑"、
            越削弱头部名次的优势。

    Returns:
        ``[(chunk_id, fused_score), ...]``，按分数降序；分数相同的按首次出现顺序。
    """
    if k <= 0:
        raise ValueError("rrf k 必须为正整数")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights 长度({len(weights)}) 必须与 rankings 数量({len(rankings)}) 一致")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    counter = 0

    for ranking, weight in zip(rankings, weights):
        if weight <= 0:
            continue
        seen_in_this_ranking: set[str] = set()
        rank = 0
        for chunk_id in ranking:
            if not chunk_id or chunk_id in seen_in_this_ranking:
                continue
            seen_in_this_ranking.add(chunk_id)
            rank += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = counter
                counter += 1

    ordered = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return ordered


def normalize_scores(values: Iterable[float]) -> list[float]:
    """min-max 归一到 [0,1]；全相同时返回 1.0（表示"并列最优"）。"""
    items = list(values)
    if not items:
        return []
    low = min(items)
    high = max(items)
    if high - low < 1e-12:
        return [1.0] * len(items)
    return [(value - low) / (high - low) for value in items]
