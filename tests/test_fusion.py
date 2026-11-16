"""RRF 融合测试（含手算对照）。"""

from __future__ import annotations

import pytest

from rag_assistant.retrieval.fusion import normalize_scores, rrf_fuse


def test_rrf_single_list_preserves_order() -> None:
    """单路时分数单调递减，顺序不变。"""
    fused = rrf_fuse([["a", "b", "c"]], k=60)
    assert [item[0] for item in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[1][1] == pytest.approx(1 / 62)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_rrf_manual_calculation() -> None:
    """两路交叉命中：手算分数与实现一致。"""
    # a: 1/61 + 1/62, b: 1/62 + 1/61 —— 完全对称，应按首次出现顺序并列
    fused = dict(rrf_fuse([["a", "b"], ["b", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_both_lists_boost() -> None:
    """被两路同时召回的文档应超过只在单路排第一的文档。"""
    fused = dict(rrf_fuse([["a", "c"], ["b", "c"]], k=60))
    assert fused["c"] > fused["a"]
    assert fused["c"] > fused["b"]


def test_rrf_weights() -> None:
    """权重为 0 的一路不贡献分数。"""
    fused = dict(rrf_fuse([["a"], ["b"]], weights=[1.0, 0.0], k=60))
    assert "a" in fused
    assert "b" not in fused

    skewed = rrf_fuse([["a"], ["b"]], weights=[3.0, 1.0], k=60)
    assert [item[0] for item in skewed] == ["a", "b"]


def test_rrf_dedupes_within_list() -> None:
    """同一路内的重复 ID 只计一次名次。"""
    fused = dict(rrf_fuse([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_rrf_ignores_empty_lists() -> None:
    fused = rrf_fuse([[], ["a"], []], k=60)
    assert [item[0] for item in fused] == ["a"]


def test_rrf_k_effect() -> None:
    """k 越大，头部优势越被削弱（分数差距变小）。"""
    small_k = dict(rrf_fuse([["a", "b"]], k=1))
    large_k = dict(rrf_fuse([["a", "b"]], k=1000))
    assert (small_k["a"] - small_k["b"]) > (large_k["a"] - large_k["b"])


def test_rrf_invalid_args() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])


def test_rrf_empty_input() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[]]) == []


def test_normalize_scores() -> None:
    assert normalize_scores([2.0, 3.0, 5.0]) == [0.0, pytest.approx(1 / 3), 1.0]
    assert normalize_scores([1.0, 1.0]) == [1.0, 1.0]
    assert normalize_scores([]) == []
