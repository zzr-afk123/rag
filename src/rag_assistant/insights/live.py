"""运维台的实时口径：在线会话、正在生成的回答、近期错误。

为什么放在内存里而不是从日志算：
- `logs/qa.jsonl` 是**事后**记录，一次提问要等回答结束才落盘，算不出"现在有几个人在用"；
- 运维台需要"此刻"的数字（在线人数、正在跑的流式回答），这类信号只在进程里有。

代价是**单进程有效**：多 worker 部署时每个进程各有一份计数，需要汇总才准。
本项目固定单 worker（见 `api/main.py` 模块说明），这里在 `snapshot()` 里显式返回
`single_process: True`，避免运维误读。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

#: 在线判定窗口：最近这么多秒内有过问答的用户算"在线"
DEFAULT_WINDOW_SECONDS = 300.0
#: 每分钟提问数用 60 秒窗口
RATE_WINDOW_SECONDS = 60.0
#: 错误环形缓冲的容量（够运维看最近发生了什么）
MAX_ERRORS = 50


@dataclass
class LiveTracker:
    """进程内的实时计数器（线程安全，开销只有一把锁）。"""

    window_seconds: float = DEFAULT_WINDOW_SECONDS
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _sessions: dict[str, float] = field(default_factory=dict, repr=False)
    _questions: deque[float] = field(default_factory=lambda: deque(maxlen=4096), repr=False)
    _latencies: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=2000), repr=False
    )
    _errors: deque[tuple[float, str, str]] = field(
        default_factory=lambda: deque(maxlen=MAX_ERRORS), repr=False
    )
    _inflight: int = 0
    _started_at: float = field(default_factory=time.time)
    _peak_online: int = 0

    # ---- 记录 ----
    def _online_locked(self, now: float) -> int:
        deadline = now - self.window_seconds
        stale = [key for key, seen in self._sessions.items() if seen < deadline]
        for key in stale:
            del self._sessions[key]
        return len(self._sessions)

    def touch(self, session_key: str) -> None:
        """标记某个会话"刚刚活跃"。问答开始与结束各调一次。"""
        now = time.time()
        with self._lock:
            self._sessions[session_key] = now
            online = self._online_locked(now)
            if online > self._peak_online:
                self._peak_online = online

    def begin(self, session_key: str) -> None:
        """一次问答开始：算在线 + 进行中 + 提问计数。"""
        now = time.time()
        with self._lock:
            self._sessions[session_key] = now
            self._questions.append(now)
            self._inflight += 1
            online = self._online_locked(now)
            if online > self._peak_online:
                self._peak_online = online

    def end(self, *, latency_ms: float | None = None) -> None:
        """一次问答结束：减进行中，顺便记一笔耗时。"""
        now = time.time()
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1
            if latency_ms is not None and latency_ms >= 0:
                self._latencies.append((now, float(latency_ms)))

    def record_error(self, kind: str, detail: str) -> None:
        with self._lock:
            self._errors.append((time.time(), str(kind)[:64], str(detail)[:400]))

    # ---- 读取 ----
    def snapshot(self) -> dict[str, object]:
        """给运维台用的一份快照。"""
        now = time.time()
        with self._lock:
            online = self._online_locked(now)
            rate_deadline = now - RATE_WINDOW_SECONDS
            recent_questions = sum(1 for ts in self._questions if ts >= rate_deadline)
            window_deadline = now - self.window_seconds
            window_questions = sum(1 for ts in self._questions if ts >= window_deadline)
            latencies = [value for ts, value in self._latencies if ts >= window_deadline]
            inflight = self._inflight
            errors = list(self._errors)[-5:]
            peak = self._peak_online
            started = self._started_at

        latencies.sort()

        def _pct(p: float) -> float:
            if not latencies:
                return 0.0
            index = min(len(latencies) - 1, max(0, int(round((len(latencies) - 1) * p))))
            return round(latencies[index], 1)

        return {
            "online": online,
            "peak_online": peak,
            "inflight": inflight,
            "questions_per_minute": recent_questions,
            "questions_window": window_questions,
            "window_seconds": int(self.window_seconds),
            "latency_ms": {
                "p50": _pct(0.5),
                "p95": _pct(0.95),
                "samples": len(latencies),
            },
            "errors_recent": [
                {"at": ts, "kind": kind, "detail": detail} for ts, kind, detail in errors
            ],
            "uptime_seconds": int(now - started),
            "single_process": True,
            "generated_at": now,
        }
