"""定时同步测试：开关、任务注册、触发计数、忙碌跳过。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from rag_assistant.api.jobs import SyncJob
from rag_assistant.config import Settings
from rag_assistant.exceptions import IndexBusyError
from rag_assistant.sync.scheduler import JOB_ID, SyncScheduler


class StubJobs:
    """替代 SyncJobManager：可控地返回任务或抛忙碌。"""

    def __init__(self, *, busy: bool = False) -> None:
        self.busy = busy
        self.calls: list[bool] = []

    def trigger(self, *, rebuild: bool = False) -> SyncJob:
        self.calls.append(rebuild)
        if self.busy:
            raise IndexBusyError("已有同步任务在执行")
        return SyncJob(job_id="job-1", rebuild=rebuild)


@pytest.fixture()
def scheduled_settings(mock_settings: Settings) -> Settings:
    """打开定时同步，但把间隔拉长（测试里不会真的等它触发）。"""
    return replace(
        mock_settings,
        sync=replace(mock_settings.sync, mode="schedule", interval_seconds=3600),
    )


def test_disabled_mode(mock_settings: Settings) -> None:
    scheduler = SyncScheduler(mock_settings, StubJobs())  # type: ignore[arg-type]
    assert scheduler.enabled is False
    assert scheduler.start() is False
    assert scheduler.running is False
    assert scheduler.status()["enabled"] is False


def test_start_registers_job(scheduled_settings: Settings) -> None:
    scheduler = SyncScheduler(scheduled_settings, StubJobs())  # type: ignore[arg-type]
    try:
        assert scheduler.start() is True
        assert scheduler.running is True
        assert scheduler.next_run_time is not None
        status = scheduler.status()
        assert status["interval_seconds"] == 3600
        assert status["next_run_at"]
        # 幂等：重复 start 不应重复注册
        assert scheduler.start() is True
    finally:
        scheduler.shutdown()
    assert scheduler.running is False
    assert scheduler.next_run_time is None


def test_trigger_counts(scheduled_settings: Settings) -> None:
    jobs = StubJobs()
    scheduler = SyncScheduler(scheduled_settings, jobs)  # type: ignore[arg-type]
    job = scheduler.trigger(reason="test")
    assert job is not None and job.job_id == "job-1"
    assert jobs.calls == [False]
    assert scheduler.status()["triggered"] == 1
    assert scheduler.status()["skipped"] == 0


def test_busy_is_skipped_not_raised(scheduled_settings: Settings) -> None:
    """定时触发撞上手动同步时应"跳过"而不是抛错（用户操作优先）。"""
    jobs = StubJobs(busy=True)
    scheduler = SyncScheduler(scheduled_settings, jobs)  # type: ignore[arg-type]
    assert scheduler.trigger(reason="test") is None
    status = scheduler.status()
    assert status["skipped"] == 1
    assert status["triggered"] == 0


def test_tick_does_not_raise(scheduled_settings: Settings) -> None:
    """APScheduler 线程里抛异常会污染调度器，_tick 必须自己吞掉。"""
    jobs = StubJobs(busy=True)
    scheduler = SyncScheduler(scheduled_settings, jobs)  # type: ignore[arg-type]
    scheduler._tick()  # 不应抛异常
    assert scheduler.status()["skipped"] == 1


def test_default_scheduler_is_lazy(scheduled_settings: Settings) -> None:
    """构造时不建 APScheduler 实例，start 才建（导入不产生线程）。"""
    scheduler = SyncScheduler(scheduled_settings, StubJobs())  # type: ignore[arg-type]
    assert scheduler._scheduler is None
    scheduler.start()
    try:
        assert scheduler._scheduler is not None
        assert scheduler._scheduler.get_job(JOB_ID) is not None
    finally:
        scheduler.shutdown()


def test_status_shape(scheduled_settings: Settings) -> None:
    scheduler = SyncScheduler(scheduled_settings, StubJobs())  # type: ignore[arg-type]
    status: dict[str, Any] = scheduler.status()
    for key in ("enabled", "running", "interval_seconds", "next_run_at", "triggered", "skipped"):
        assert key in status
