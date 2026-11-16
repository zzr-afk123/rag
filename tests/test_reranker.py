"""Reranker 测试（mock 实现 + 工厂 + 归一化）。"""

from __future__ import annotations

import pytest

from rag_assistant.config import RerankerConfig
from rag_assistant.retrieval.reranker import (
    CrossEncoderReranker,
    MockReranker,
    get_reranker,
    _lcs_ratio,
    _sigmoid,
)


def test_mock_reranker_shape_and_range() -> None:
    reranker = MockReranker()
    texts = ["学生休学须经教务处审批", "在读证明可线上办理", "科研经费报销标准"]
    scores = reranker.score("休学怎么办", texts)
    assert len(scores) == len(texts)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert reranker.score("问题", []) == []


def test_mock_reranker_ranks_relevant_first() -> None:
    """与 query 字面更接近的排前面。"""
    reranker = MockReranker()
    texts = ["科研经费差旅费报销须在 15 个工作日内办理", "学生休学办理流程：本人申请，二级学院审核"]
    scores = reranker.score("休学办理流程", texts)
    assert scores[1] > scores[0]


def test_mock_reranker_exact_substring() -> None:
    """整串命中给高分。"""
    reranker = MockReranker()
    scores = reranker.score("在读证明怎么开", ["在读证明怎么开：线上或线下均可"])
    assert scores[0] >= 0.9


def test_mock_reranker_deterministic() -> None:
    reranker = MockReranker()
    first = reranker.score("休学办理", ["休学办理流程说明"])
    second = reranker.score("休学办理", ["休学办理流程说明"])
    assert first == second
    assert reranker.is_mock is True


def test_sigmoid_range() -> None:
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(10.0) > 0.99
    assert _sigmoid(-10.0) < 0.01
    assert 0.0 < _sigmoid(-3.0) < 0.05


def test_lcs_ratio() -> None:
    assert _lcs_ratio("休学办理", "学生休学办理流程") == pytest.approx(1.0)
    assert _lcs_ratio("休学办理", "在读证明") == 0.0
    assert _lcs_ratio("", "abc") == 0.0


def test_factory_disabled_returns_none() -> None:
    assert get_reranker(RerankerConfig(enabled=False)) is None
    assert get_reranker(RerankerConfig(provider="none")) is None


def test_factory_mock() -> None:
    reranker = get_reranker(RerankerConfig(provider="mock"))
    assert isinstance(reranker, MockReranker)


def test_factory_bge_is_lazy() -> None:
    """构造 bge 重排器不加载模型（避免测试环境下载 GB 级权重）。"""
    reranker = get_reranker(RerankerConfig(provider="bge", model="BAAI/bge-reranker-v2-m3"))
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker._model is None


def test_factory_unknown_provider() -> None:
    with pytest.raises(ValueError):
        get_reranker(RerankerConfig(provider="nope"))
