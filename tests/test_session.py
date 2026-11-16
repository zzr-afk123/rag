"""会话层测试：多轮记录、会话隔离、摘要压缩、SQLite 持久化。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from rag_assistant.config import SessionConfig, Settings
from rag_assistant.llm import MockLLM
from rag_assistant.llm.base import LLMClient, LLMMessage, LLMResponse, LLMUsage
from rag_assistant.session.store import (
    HistoryTurn,
    MemorySessionStore,
    SessionManager,
    SqliteSessionStore,
    build_store,
)


def turn(question: str, answer: str = "回答", refused: bool = False) -> HistoryTurn:
    return HistoryTurn(question=question, answer=answer, refused=refused)


class FailingLLM(LLMClient):
    model = "failing"

    def complete(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> LLMResponse:
        raise RuntimeError("摘要服务不可用")

    def stream(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None):
        raise RuntimeError("摘要服务不可用")
        yield ""  # pragma: no cover


# --------------------------------------------------------------------------
# 基本行为
# --------------------------------------------------------------------------
def test_get_creates_session() -> None:
    manager = SessionManager(SessionConfig(), MockLLM())
    session = manager.get("s1")
    assert session.session_id == "s1"
    assert session.turns == []


def test_record_appends_turn() -> None:
    manager = SessionManager(SessionConfig(), MockLLM())
    manager.record("s1", turn("休学多久"))
    manager.record("s1", turn("那复学呢"))
    turns = manager.list_turns("s1")
    assert [item.question for item in turns] == ["休学多久", "那复学呢"]


def test_sessions_are_isolated() -> None:
    """会话之间互不干扰（需求 4）。"""
    manager = SessionManager(SessionConfig(), MockLLM())
    manager.record("a", turn("A 的问题"))
    manager.record("b", turn("B 的问题"))

    assert [t.question for t in manager.list_turns("a")] == ["A 的问题"]
    assert [t.question for t in manager.list_turns("b")] == ["B 的问题"]
    assert manager.history_text("a").count("B 的问题") == 0


def test_empty_session_id_rejected() -> None:
    manager = SessionManager(SessionConfig(), MockLLM())
    with pytest.raises(ValueError):
        manager.get("   ")


def test_reset_clears_session() -> None:
    manager = SessionManager(SessionConfig(), MockLLM())
    manager.record("s1", turn("问题"))
    manager.reset("s1")
    assert manager.list_turns("s1") == []
    assert manager.history_text("s1") == ""


# --------------------------------------------------------------------------
# 历史渲染
# --------------------------------------------------------------------------
def test_history_text_format() -> None:
    manager = SessionManager(SessionConfig(max_turns=6), MockLLM())
    manager.record("s1", turn("休学多久", "休学期一般以一年为期。"))
    text = manager.history_text("s1")
    assert "用户：休学多久" in text
    assert "助手：休学期一般以一年为期。" in text


def test_history_long_answer_truncated() -> None:
    manager = SessionManager(SessionConfig(), MockLLM())
    manager.record("s1", turn("问题", "很长的回答" * 100))
    text = manager.history_text("s1")
    assert "…" in text
    assert len(text) < 400


# --------------------------------------------------------------------------
# 压缩
# --------------------------------------------------------------------------
def test_compress_when_threshold_exceeded() -> None:
    """超过 summarize_after 轮时压缩最老的部分，只保留 max_turns 轮。"""
    config = SessionConfig(max_turns=2, summarize_after=2)
    llm = MockLLM()
    manager = SessionManager(config, llm)

    for index in range(4):
        manager.record("s1", turn(f"第{index}个问题", f"第{index}个回答"))

    session = manager.get("s1")
    assert len(session.turns) == 2
    assert session.summary
    assert "summary" in llm.calls  # 确实调用了摘要任务
    assert [t.question for t in session.turns] == ["第2个问题", "第3个问题"]


def test_compress_failure_falls_back() -> None:
    """压缩失败不能把整轮问答带崩：降级为硬截断。"""
    config = SessionConfig(max_turns=1, summarize_after=1)
    manager = SessionManager(config, FailingLLM())

    manager.record("s1", turn("问题一", "回答一"))
    manager.record("s1", turn("问题二", "回答二"))

    session = manager.get("s1")
    assert len(session.turns) == 1
    assert session.summary  # 仍生成了降级摘要


def test_no_compress_below_threshold() -> None:
    config = SessionConfig(max_turns=3, summarize_after=5)
    manager = SessionManager(config, MockLLM())
    manager.record("s1", turn("问题一"))
    assert manager.compress_if_needed("s1") is False
    assert manager.get("s1").summary == ""


# --------------------------------------------------------------------------
# SQLite 存储
# --------------------------------------------------------------------------
def test_sqlite_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SqliteSessionStore(path)
    manager = SessionManager(SessionConfig(), MockLLM(), store=store)
    manager.record("s1", turn("休学多久", "一年为期"))

    reopened = SqliteSessionStore(path)
    manager2 = SessionManager(SessionConfig(), MockLLM(), store=reopened)
    turns = manager2.list_turns("s1")
    assert len(turns) == 1
    assert turns[0].question == "休学多久"
    store.close()
    reopened.close()


def test_sqlite_delete(tmp_path: Path) -> None:
    store = SqliteSessionStore(tmp_path / "s.db")
    manager = SessionManager(SessionConfig(), MockLLM(), store=store)
    manager.record("s1", turn("问题"))
    manager.reset("s1")
    assert store.load("s1") is None
    store.close()


def test_build_store_by_config(tmp_path: Path) -> None:
    assert isinstance(build_store(SessionConfig(store="memory"), tmp_path), MemorySessionStore)
    sqlite_store = build_store(SessionConfig(store="sqlite", sqlite_path="data/s.db"), tmp_path)
    assert isinstance(sqlite_store, SqliteSessionStore)
    assert (tmp_path / "data" / "s.db").exists()
    sqlite_store.close()


def test_history_text_includes_summary(mock_settings: Settings) -> None:
    """摘要要出现在给 prompt 的历史里（否则压缩等于丢上下文）。"""
    settings = replace(mock_settings, session=replace(mock_settings.session, max_turns=1, summarize_after=1))
    manager = SessionManager.from_settings(settings, MockLLM())
    manager.record("s1", turn("休学半年可以吗", "可以，但需审批"))
    manager.record("s1", turn("那复学呢", "需康复证明"))

    text = manager.history_text("s1")
    assert "[早期对话摘要]" in text
    assert "休学半年可以吗" in text  # 摘要或最近轮次里应有这个实体
