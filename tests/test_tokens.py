"""token 估算与句子切分测试。"""

from __future__ import annotations

from rag_assistant.tokens import (
    count_tokens,
    heuristic_token_count,
    split_sentences,
    tail_tokens,
    truncate_to_tokens,
)


def test_count_tokens_positive() -> None:
    assert count_tokens("") == 0
    assert count_tokens("学生注册") > 0
    assert count_tokens("The registrar office") > 0


def test_heuristic_scales_with_length() -> None:
    short = heuristic_token_count("学生注册")
    long = heuristic_token_count("学生注册" * 10 + "，逾期不予补办。")
    assert long > short


def test_split_sentences_basic() -> None:
    sentences = split_sentences("第一条 目的。第二条 范围！第三条 时间？")
    assert len(sentences) == 3
    assert sentences[0].startswith("第一条")


def test_split_sentences_keeps_decimals() -> None:
    """小数与编号不被当成句末断开。"""
    sentences = split_sentences("第3.5条规定，报销限额为1.2万元。")
    assert len(sentences) == 1
    assert "3.5" in sentences[0]


def test_split_sentences_newline() -> None:
    sentences = split_sentences("第一行\n第二行")
    assert sentences == ["第一行", "第二行"]


def test_tail_tokens_is_suffix() -> None:
    text = "第一条 内容甲。第二条 内容乙。第三条 内容丙。"
    tail = tail_tokens(text, 12)
    assert tail
    assert text.endswith(tail)
    assert len(tail) < len(text)


def test_tail_tokens_zero() -> None:
    assert tail_tokens("内容", 0) == ""
    assert tail_tokens("", 10) == ""


def test_truncate_to_tokens() -> None:
    text = "学生应当在规定时间内完成注册。" * 20
    truncated = truncate_to_tokens(text, 30)
    assert count_tokens(truncated) <= 30
    assert truncate_to_tokens("短文本", 30) == "短文本"
