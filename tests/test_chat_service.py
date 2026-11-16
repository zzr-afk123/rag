"""对话服务端到端测试：多轮、会话隔离、拒答记录、流式、日志。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.kb import KnowledgeBase
from rag_assistant.logging_setup import setup_logging


@pytest.fixture()
def service(mock_settings: Settings, sample_raw_dir: Path):
    """建好索引的对话服务（mock 链路：哈希向量 + 内存库 + 词面重排）。"""
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    chat = ChatService(mock_settings, kb=kb)
    yield chat
    chat.close()


def test_single_turn_with_citations(service: ChatService) -> None:
    # mock 分数绝对值偏低，用 threshold=0 保证进入生成阶段（阈值判定另有测试）
    reply = service.chat("s1", "在读证明在哪里办、多久能拿到", threshold=0.0)

    assert reply.refused is False
    assert reply.citations, "回答必须带引用"
    assert reply.citations[0].doc_name == "办事指南-在读证明.txt"
    assert "引用来源" in reply.answer
    assert reply.rewritten_query
    assert reply.latency_ms > 0
    assert reply.usage_prompt + reply.usage_completion > 0

    turns = service.sessions.list_turns("s1")
    assert len(turns) == 1
    assert turns[0].question.startswith("在读证明")


def test_multi_turn_uses_history(service: ChatService) -> None:
    """第二轮问"那复学呢"，改写必须带上第一轮的上下文。"""
    first = service.chat("s1", "休学最长可以多久", threshold=0.0)
    assert first.refused is False

    second = service.chat("s1", "那复学呢", threshold=0.0)
    assert second.rewritten_query != "那复学呢", "改写应当利用历史消解指代"
    assert "休学" in second.rewritten_query
    assert len(service.sessions.list_turns("s1")) == 2


def test_sessions_are_isolated(service: ChatService) -> None:
    service.chat("a", "休学最长可以多久", threshold=0.0)
    reply_b = service.chat("b", "在读证明怎么办", threshold=0.0)

    assert "休学" not in reply_b.rewritten_query
    assert len(service.sessions.list_turns("a")) == 1
    assert len(service.sessions.list_turns("b")) == 1


def test_session_owner_isolates_same_session_id(service: ChatService) -> None:
    """session_id 由浏览器生成，可猜可抄；服务端必须按登录身份再分一层。"""
    service.chat("shared", "休学最长可以多久", threshold=0.0, owner="20230001")
    # 别人拿着同一个 session_id 提问，不能带上前面那个人的上下文
    reply = service.chat("shared", "那复学呢", threshold=0.0, owner="20230002")

    assert "休学" not in reply.rewritten_query
    assert len(service.sessions.list_turns("20230001::shared")) == 1
    assert len(service.sessions.list_turns("20230002::shared")) == 1

    # 同一个人刷新页面（session_id 不变）仍能续上
    again = service.chat("shared", "那复学呢", threshold=0.0, owner="20230001")
    assert len(service.sessions.list_turns("20230001::shared")) == 2
    assert again.session_id == "shared"


def test_reset_only_clears_own_session(service: ChatService) -> None:
    service.chat("s", "休学最长可以多久", threshold=0.0, owner="me")
    service.chat("s", "在读证明怎么办", threshold=0.0, owner="other")

    service.reset("s", owner="me")

    assert service.sessions.list_turns("me::s") == []
    assert len(service.sessions.list_turns("other::s")) == 1


def test_refusal_is_recorded(service: ChatService) -> None:
    reply = service.chat("s1", "图书馆几点开门", threshold=0.5)

    assert reply.refused is True
    assert reply.citations == []
    assert reply.refusal_reason in {"below_threshold", "no_candidates", "off_topic", "llm_refused", "citation_invalid"}
    turns = service.sessions.list_turns("s1")
    assert turns[0].refused is True


def test_reset_session(service: ChatService) -> None:
    service.chat("s1", "休学最长可以多久", threshold=0.0)
    service.reset("s1")
    assert service.sessions.list_turns("s1") == []

    # 重置后第一轮不应再受旧上下文影响：改写里不该出现"休学"
    reply = service.chat("s1", "那复学呢", threshold=0.0)
    assert "休学" not in reply.rewritten_query
    assert "复学" in reply.rewritten_query


def test_stream_events_and_history(service: ChatService) -> None:
    events = list(service.chat_stream("s1", "在读证明怎么开", threshold=0.0))
    kinds = [event["type"] for event in events]

    assert kinds[0] == "meta"
    assert "token" in kinds
    assert kinds[-1] in {"done", "retract"}

    turns = service.sessions.list_turns("s1")
    assert len(turns) == 1
    assert turns[0].answer


def test_qa_log_written(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """每次问答写一行 JSONL（需求 8：可观测性）。"""
    setup_logging(mock_settings, force=True)
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    chat = ChatService(mock_settings, kb=kb)
    try:
        chat.chat("log-test", "在读证明怎么开", threshold=0.0)
    finally:
        chat.close()

    log_file = mock_settings.log_dir / mock_settings.logging.qa_log
    assert log_file.exists()
    lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    match = [record for record in records if record.get("session_id") == "log-test"]
    assert match, records[-3:]
    record = match[-1]
    assert record["question"] == "在读证明怎么开"
    assert record["refused"] is False
    assert "latency_ms" in record


def test_usage_and_threshold_diagnostics(service: ChatService) -> None:
    reply = service.chat("s1", "会议费由谁审批", threshold=0.0)
    assert reply.refused is False
    assert reply.closest_docs
    assert reply.closest_docs[0]["doc_name"]
