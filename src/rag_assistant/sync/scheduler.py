"""定时同步（APScheduler）。

职责很窄：**按间隔触发一次同步**，不直接碰索引。真正的写入由
``KnowledgeBase.sync()`` 在写锁里完成，任务状态由 ``SyncJobManager`` 记录。

两个关键细节：
1. ``max_instances=1`` + ``coalesce=True``：上一次还没跑完就跳过本 tick，
   避免堆积（比"排队等第 5 次同步"合理）。
2. 触发时如果已有任务在跑（比如用户刚点了手动同步），**记为 skipped 而不是报错**：
   定时任务打扰用户的操作没有意义。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..api.jobs import SyncJob, SyncJobManager
from ..config import Settings
from ..exceptions import IndexBusyError
from ..logging_setup import get_logger

logger = get_logger("sync.scheduler")

JOB_ID = "kb-auto-sync"


class SyncScheduler:
    """把 ``sync.interval_seconds`` 变成实际的定时任务。"""

    def __init__(
        self,
        settings: Settings,
        jobs: SyncJobManager,
        *,
        scheduler: Any | None = None,
    ) -> None:
        self._settings = settings
        self._jobs = jobs
        self._scheduler = scheduler
        self._started = False
        self._skipped = 0
        self._triggered = 0

    # ---- 生命周期 ----
    @property
    def enabled(self) -> bool:
        return self._settings.sync.mode == "schedule"

    @property
    def running(self) -> bool:
        return self._started and self._scheduler is not None and self._scheduler.running

    def start(self) -> bool:
        """启动定时任务；``mode != schedule`` 时直接返回 False。"""
        if not self.enabled:
            logger.info("sync.mode=%s，不启动定时同步", self._settings.sync.mode)
            return False
        if self._started:
            return True

        if self._scheduler is None:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()

        self._scheduler.add_job(
            self._tick,
            trigger="interval",
            seconds=self._settings.sync.interval_seconds,
            id=JOB_ID,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=max(30, self._settings.sync.interval_seconds),
        )
        self._scheduler.start()
        self._started = True
        logger.info("定时同步已启动：每 %ss 扫描一次", self._settings.sync.interval_seconds)
        return True

    def shutdown(self) -> None:
        if self._scheduler is not None and self._started:
            try:
                self._scheduler.remove_job(JOB_ID)
            except Exception:  # noqa: BLE001 - 任务可能已被移除
                pass
            self._scheduler.shutdown(wait=False)
            logger.info("定时同步已停止")
        self._started = False

    # ---- 触发 ----
    def _tick(self) -> None:
        """由 APScheduler 线程调用。"""
        self.trigger(reason="scheduler")

    def trigger(self, *, reason: str = "scheduler", rebuild: bool = False) -> SyncJob | None:
        """触发一次同步；已有任务在跑时返回 None 并记录 skipped。"""
        try:
            job = self._jobs.trigger(rebuild=rebuild)
            self._triggered += 1
            logger.info("同步已触发（%s）：job=%s", reason, job.job_id)
            return job
        except IndexBusyError:
            self._skipped += 1
            logger.info("同步跳过（%s）：已有任务在执行", reason)
            return None
        except Exception:  # noqa: BLE001 - 定时线程不能抛崩
            logger.exception("触发同步失败（%s）", reason)
            return None

    # ---- 状态 ----
    @property
    def next_run_time(self) -> datetime | None:
        if not self.running or self._scheduler is None:
            return None
        job = self._scheduler.get_job(JOB_ID)
        return getattr(job, "next_run_time", None) if job else None

    def status(self) -> dict[str, Any]:
        moment = self.next_run_time
        return {
            "enabled": self.enabled,
            "running": self.running,
            "interval_seconds": self._settings.sync.interval_seconds,
            "next_run_at": moment.isoformat() if moment else None,
            "triggered": self._triggered,
            "skipped": self._skipped,
        }


def build_scheduler(settings: Settings, jobs: SyncJobManager) -> SyncScheduler:
    return SyncScheduler(settings, jobs)


__all__ = ["JOB_ID", "SyncScheduler", "build_scheduler"]
