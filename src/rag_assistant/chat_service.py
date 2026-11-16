"""对话服务：把知识库、会话、改写、生成串成"一次问答"。

为什么单独一层而不是塞进 API：
- 阶段 D 的 FastAPI、CLI 演示脚本、阶段 F 的评测脚本都需要"给问题拿答案"，
  但都不该关心 HTTP 或 SSE；
- 会话写入、日志落盘、usage 汇总这些"每次问答都要做"的副作用集中在这里，
  避免在路由里散落。

日志口径：写进 ``logs/qa.jsonl`` 的字段是固定的（问题、改写、召回 ID、耗时、
是否拒答、token 用量）。**不记录完整答案正文**，避免把制度原文与用户问题
长期堆积在日志里。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from .config import Settings, get_settings
from .departments import DepartmentDirectory
from .generation.answerer import Answerer, AnswerResult
from .generation.citation import Citation
from .generation.rewriter import QueryRewriter
from .insights.live import LiveTracker
from .kb import KnowledgeBase
from .llm import LLMClient, get_llm
from .logging_setup import get_logger, log_qa
from .session.store import HistoryTurn, SessionManager

logger = get_logger("chat")


@dataclass
class ChatReply:
    """对外的一轮问答结果。"""

    session_id: str
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    rewritten_query: str = ""
    extra_queries: list[str] = field(default_factory=list)
    closest_docs: list[dict[str, object]] = field(default_factory=list)
    #: 命中的相关部门（含联系方式），前端渲染成卡片；不进入 answer 正文
    contacts: list[dict[str, object]] = field(default_factory=list)
    latency_ms: float = 0.0
    usage_prompt: int = 0
    usage_completion: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "rewritten_query": self.rewritten_query,
            "extra_queries": self.extra_queries,
            "closest_docs": self.closest_docs,
            "contacts": self.contacts,
            "latency_ms": round(self.latency_ms, 1),
            "usage": {"prompt_tokens": self.usage_prompt, "completion_tokens": self.usage_completion},
        }


class ChatService:
    """问答门面。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        kb: KnowledgeBase | None = None,
        llm: LLMClient | None = None,
        sessions: SessionManager | None = None,
        live: LiveTracker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.kb = kb or KnowledgeBase(self.settings)
        self.llm = llm or get_llm(self.settings)
        self.sessions = sessions or SessionManager.from_settings(self.settings, self.llm)
        # 运维台的"此刻"数字：在线人数、正在生成的回答、最近的耗时
        self.live = live or LiveTracker()
        self.rewriter = QueryRewriter(self.settings, self.llm)
        self.departments = DepartmentDirectory.from_settings(self.settings)
        self.answerer = Answerer(self.settings, self.kb.retriever, self.llm, self.rewriter)

    # ---- 同步问答 ----
    def chat(
        self,
        session_id: str,
        question: str,
        *,
        threshold: float | None = None,
        owner: str | None = None,
    ) -> ChatReply:
        """一问一答（阻塞）。``owner`` 非空时会话按用户隔离。"""
        started = time.perf_counter()
        key = self._session_key(session_id, owner)
        history = self.sessions.history_text(key)
        self.live.begin(key)
        try:
            result = self.answerer.answer(question, history_text=history, threshold=threshold)
        except Exception as exc:  # noqa: BLE001 - 记一笔再抛给上层做异常映射
            self.live.record_error(type(exc).__name__, question[:200])
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            self.live.end(latency_ms=latency_ms)

        self.sessions.record(
            key,
            HistoryTurn(
                question=question,
                answer=result.answer,
                rewritten_query=result.rewritten_query,
                refused=result.refused,
            ),
        )
        self._log(key, result, latency_ms)

        return ChatReply(
            session_id=session_id,
            question=question,
            answer=result.answer,
            citations=result.citations,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            rewritten_query=result.rewritten_query,
            extra_queries=result.extra_queries,
            closest_docs=result.closest_docs,
            contacts=result.contacts,
            latency_ms=latency_ms,
            usage_prompt=result.usage.prompt_tokens,
            usage_completion=result.usage.completion_tokens,
        )

    # ---- 流式问答 ----
    def chat_stream(
        self,
        session_id: str,
        question: str,
        *,
        threshold: float | None = None,
        owner: str | None = None,
    ) -> Iterator[dict[str, object]]:
        """流式问答：原样转发 ``Answerer.stream`` 的事件，并在结束时记录会话与日志。"""
        started = time.perf_counter()
        key = self._session_key(session_id, owner)
        history = self.sessions.history_text(key)
        answer_text = ""
        refused = False
        refusal_reason: str | None = None
        rewritten = ""
        chunk_ids: list[str] = []
        contacts: list[dict[str, object]] = []
        support_rate: float | None = None

        self.live.begin(key)
        try:
            for event in self.answerer.stream(question, history_text=history, threshold=threshold):
                kind = event.get("type")
                if kind == "meta":
                    rewritten = str(event.get("rewritten_query", ""))
                    chunk_ids = [str(item) for item in event.get("retrieved_ids", [])]
                elif kind in {"done", "retract"}:
                    answer_text = str(event.get("answer", ""))
                    refused = bool(event.get("refused", kind == "retract"))
                    refusal_reason = event.get("refusal_reason") if kind == "done" else event.get("reason")
                    contacts = list(event.get("contacts") or [])
                    claims = event.get("claims")
                    if isinstance(claims, dict):
                        value = claims.get("support_rate")
                        if isinstance(value, (int, float)):
                            support_rate = float(value)
                yield event
        finally:
            latency_ms = (time.perf_counter() - started) * 1000
            self.live.end(latency_ms=latency_ms)
            if answer_text:
                self.sessions.record(
                    key,
                    HistoryTurn(
                        question=question,
                        answer=answer_text,
                        rewritten_query=rewritten,
                        refused=refused,
                    ),
                )
                log_qa(
                    {
                        "session_id": key,
                        "question": question,
                        "rewritten_query": rewritten,
                        "retrieved_ids": chunk_ids,
                        "latency_ms": round(latency_ms, 1),
                        "refused": refused,
                        "refusal_reason": refusal_reason,
                        "departments": [str(item.get("id")) for item in contacts],
                        "claims_support": support_rate,
                        "stream": True,
                    }
                )

    # ---- 辅助 ----
    @staticmethod
    def _session_key(session_id: str, owner: str | None) -> str:
        """会话按用户隔离。

        session_id 是浏览器自己生成的，可猜也可被抄走；在服务端拼上登录身份之后，
        同一个人刷新页面仍能续上上下文，换个人拿到同一个 id 也读不到别人的历史。
        """
        return f"{owner}::{session_id}" if owner else session_id

    @staticmethod
    def _log(session_id: str, result: AnswerResult, latency_ms: float) -> None:
        retrieval = result.diagnostics.get("retrieval") or {}
        claims = result.diagnostics.get("claims") or {}
        log_qa(
            {
                "session_id": session_id,
                "question": result.question,
                "rewritten_query": result.rewritten_query,
                "extra_queries": result.extra_queries,
                "retrieved_ids": [citation.chunk_id for citation in result.citations],
                "latency_ms": round(latency_ms, 1),
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "departments": [str(item.get("id")) for item in result.contacts],
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "best_score": retrieval.get("best_score"),
                "claims_support": claims.get("support_rate"),
            }
        )

    def reset(self, session_id: str, *, owner: str | None = None) -> None:
        """清空某个会话的上下文。"""
        self.sessions.reset(self._session_key(session_id, owner))

    def close(self) -> None:
        """释放底层资源（会话存储、向量库）。"""
        store = self.sessions.store
        close = getattr(store, "close", None)
        if callable(close):
            close()
        self.kb.close()
