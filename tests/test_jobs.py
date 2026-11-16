"""后台同步任务管理测试：单飞约束、状态流转、失败隔离。"""

from __future__ import annotations

import time

import pytest

from rag_assistant.api.jobs import SyncJobManager
from rag_assistant.exceptions import IndexBusyError
from rag_assistant.kb import SyncReport


class SlowKB:
    """假的 KnowledgeBase：同步耗时可控，便于断言"任务正在跑"。"""

    def __init__(self, *, delay: float = 0.15, fail: bool = False, with_failures: bool = False) -> None:
        self.delay = delay
        self.fail = fail
        self.with_failures = with_failures
        self.calls: list[bool] = []

    def sync(self, *, rebuild: bool = False, limit: int | None = None) -> SyncReport:
        self.calls.append(rebuild)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("磁盘写满了")
        return SyncReport(
            added=2,
            chunks_written=10,
            failed=[{"file": "bad.docx", "stage": "ingest", "error": "损坏"}] if self.with_failures else [],
        )


def test_trigger_and_wait() -> None:
    kb = SlowKB(delay=0.05)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]

    job = manager.trigger(rebuild=True)
    assert job.job_id
    assert manager.is_running is True

    finished = manager.wait(timeout=5)
    assert finished is not None
    assert finished.status == "success"
    assert finished.report["added"] == 2
    assert finished.duration_ms is not None
    assert manager.is_running is False
    assert kb.calls == [True]


def test_single_flight() -> None:
    """并发触发必须被拒绝（否则向量库/清单会互相覆盖）。"""
    kb = SlowKB(delay=0.2)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]
    manager.trigger()
    with pytest.raises(IndexBusyError, match="已有同步任务"):
        manager.trigger()
    manager.wait(timeout=5)

    # 结束后可以再次触发
    manager.trigger()
    manager.wait(timeout=5)
    assert len(kb.calls) == 2


def test_failure_reported() -> None:
    kb = SlowKB(delay=0.01, fail=True)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]
    manager.trigger()
    job = manager.wait(timeout=5)

    assert job is not None
    assert job.status == "failed"
    assert "磁盘写满了" in (job.error or "")
    assert manager.last_error


def test_partial_failures_mark_failed() -> None:
    """单文件失败时任务标记为 failed（但报告里仍保留成功部分）。"""
    kb = SlowKB(delay=0.01, with_failures=True)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]
    job = manager.run_inline()

    assert job.status == "failed"
    assert job.report["chunks_written"] == 10
    assert "1 个文件同步失败" in (job.error or "")


def test_run_inline_is_synchronous() -> None:
    kb = SlowKB(delay=0.01)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]
    job = manager.run_inline(rebuild=False)
    assert job.finished_at is not None
    assert job.status == "success"


def test_run_inline_respects_single_flight() -> None:
    kb = SlowKB(delay=0.2)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]
    manager.trigger()
    with pytest.raises(IndexBusyError):
        manager.run_inline()
    manager.wait(timeout=5)


def test_to_dict_shape() -> None:
    kb = SlowKB(delay=0.01)
    manager = SyncJobManager(kb)  # type: ignore[arg-type]
    job = manager.run_inline()
    payload = job.to_dict()
    for key in ("job_id", "status", "rebuild", "started_at", "finished_at", "duration_ms", "report", "error"):
        assert key in payload
