"""会话存储与上下文压缩。

需求：``session_id`` 管理多轮会话，保存最近 N 轮，超出后做摘要压缩；会话之间互不干扰。

设计取舍：
- **存储**：默认内存（重启即失效），可切 SQLite。会话不是关键数据，
  引入 Redis/DB 服务对当前规模是过度设计。
- **压缩时机**：``len(turns) > summarize_after`` 时，把最老的若干轮折叠进
  ``summary``，只保留最近 ``max_turns`` 轮原文。压缩失败（LLM 报错）时
  **降级为硬截断**而不是抛异常——丢一点上下文，总比整轮问答失败好。
- **隐私**：会话内容会进日志（可选关闭），因此日志里只记问题与耗时，
  不记完整回答正文（见 ``log_qa`` 的调用点）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..config import SessionConfig, Settings
from ..exceptions import LLMError
from ..llm.base import LLMClient
from ..llm.prompts import summary_prompt
from ..logging_setup import get_logger
from ..tokens import truncate_to_tokens

logger = get_logger("session")

#: 写进 prompt 时，单条回答最多保留的字符数（避免历史吃掉上下文预算）
_ANSWER_PREVIEW_CHARS = 120


@dataclass
class HistoryTurn:
    """一轮问答。"""

    question: str
    answer: str
    rewritten_query: str = ""
    refused: bool = False
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "rewritten_query": self.rewritten_query,
            "refused": self.refused,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoryTurn":
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            rewritten_query=str(data.get("rewritten_query", "")),
            refused=bool(data.get("refused", False)),
            ts=float(data.get("ts", 0.0)),
        )

    def render(self, *, max_chars: int = _ANSWER_PREVIEW_CHARS) -> str:
        answer = self.answer.replace("\n", " ")
        if len(answer) > max_chars:
            answer = answer[:max_chars] + "…"
        return f"用户：{self.question}\n助手：{answer}"


@dataclass
class Session:
    """一个会话。"""

    session_id: str
    turns: list[HistoryTurn] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "turns": [turn.to_dict() for turn in self.turns],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=str(data.get("session_id", "")),
            turns=[HistoryTurn.from_dict(item) for item in data.get("turns", [])],
            summary=str(data.get("summary", "")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


# --------------------------------------------------------------------------
# 存储实现
# --------------------------------------------------------------------------
class SessionStore(ABC):
    """会话存储接口。"""

    @abstractmethod
    def load(self, session_id: str) -> Session | None:
        """读取会话；不存在返回 None。"""

    @abstractmethod
    def save(self, session: Session) -> None:
        """写入/覆盖会话。"""

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """删除会话。"""


class MemorySessionStore(SessionStore):
    """进程内存储（默认）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return Session.from_dict(session.to_dict()) if session else None

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = Session.from_dict(session.to_dict())

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    @property
    def size(self) -> int:
        return len(self._sessions)


class SqliteSessionStore(SessionStore):
    """SQLite 存储（单文件、可跨进程、无需额外服务）。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                payload    TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def load(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if not row:
            return None
        try:
            return Session.from_dict(json.loads(row[0]))
        except json.JSONDecodeError:  # pragma: no cover - 数据损坏时按新会话处理
            logger.warning("会话数据损坏，按空会话处理：%s", session_id)
            return None

    def save(self, session: Session) -> None:
        payload = json.dumps(session.to_dict(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, payload, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (session.session_id, payload, session.updated_at),
            )
            self._conn.commit()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def build_store(config: SessionConfig, root: Path) -> SessionStore:
    """按配置创建存储。"""
    if config.store == "sqlite":
        return SqliteSessionStore(root / config.sqlite_path)
    return MemorySessionStore()


# --------------------------------------------------------------------------
# 管理器
# --------------------------------------------------------------------------
class SessionManager:
    """负责"取会话、记一轮、压缩历史"这三件事。"""

    def __init__(self, config: SessionConfig, llm: LLMClient, store: SessionStore | None = None) -> None:
        self._config = config
        self._llm = llm
        self._store = store or MemorySessionStore()

    @classmethod
    def from_settings(cls, settings: Settings, llm: LLMClient, store: SessionStore | None = None) -> "SessionManager":
        return cls(settings.session, llm, store or build_store(settings.session, settings.root))

    # ---- 基本操作 ----
    def get(self, session_id: str) -> Session:
        """取会话，不存在则创建（不落盘，直到有内容）。"""
        if not session_id or not session_id.strip():
            raise ValueError("session_id 不能为空")
        session = self._store.load(session_id)
        if session is None:
            session = Session(session_id=session_id)
        return session

    def record(self, session_id: str, turn: HistoryTurn) -> Session:
        """记录一轮问答，并在超限时触发压缩。"""
        session = self.get(session_id)
        session.turns.append(turn)
        session.updated_at = time.time()
        self._store.save(session)
        self.compress_if_needed(session_id)
        return self.get(session_id)

    def reset(self, session_id: str) -> None:
        self._store.delete(session_id)

    # ---- 上下文 ----
    def history_text(self, session_id: str, *, max_tokens: int = 600) -> str:
        """渲染给 prompt 的历史文本（摘要 + 最近 N 轮）。"""
        session = self.get(session_id)
        if not session.turns and not session.summary:
            return ""
        parts: list[str] = []
        if session.summary:
            parts.append(f"[早期对话摘要] {session.summary}")
        recent = session.turns[-self._config.max_turns :]
        parts.extend(turn.render() for turn in recent)
        text = "\n".join(parts)
        # 兜底：历史太长就截断（宁可丢最老的，也不要撑爆上下文）
        return truncate_to_tokens(text, max_tokens) if text else ""

    # ---- 压缩 ----
    def compress_if_needed(self, session_id: str) -> bool:
        """超过阈值时把最老的若干轮折叠进摘要。返回是否发生了压缩。"""
        session = self.get(session_id)
        limit = max(1, self._config.summarize_after)
        if len(session.turns) <= limit:
            return False

        keep = max(1, self._config.max_turns)
        overflow = session.turns[:-keep]
        if not overflow:
            return False

        turns_text = "\n".join(turn.render(max_chars=200) for turn in overflow)
        try:
            response = self._llm.complete(summary_prompt(session.summary, turns_text))
            summary = response.text.strip()
            if not summary:
                raise LLMError("摘要结果为空")
        except Exception as exc:  # noqa: BLE001 - 压缩失败必须降级，不能中断问答
            logger.warning("会话摘要压缩失败，降级为硬截断：%s", exc)
            summary = (session.summary + " " + " ".join(turns_text.split()))[:300].strip()

        session.summary = summary
        session.turns = session.turns[-keep:]
        session.updated_at = time.time()
        self._store.save(session)
        logger.info("会话 %s 已压缩历史，摘要 %d 字，保留 %d 轮", session_id, len(summary), len(session.turns))
        return True

    @property
    def store(self) -> SessionStore:
        return self._store

    def list_turns(self, session_id: str) -> Sequence[HistoryTurn]:
        return list(self.get(session_id).turns)
