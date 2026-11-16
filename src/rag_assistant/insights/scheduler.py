"""周报定时任务（APScheduler cron）。

与 `sync.scheduler` 一样的取舍：`max_instances=1` + `coalesce=True`，
错过了就跳过（补发的意义不大，看板随时能看到最新数据）。

默认不启动：`report.enabled=false`。要启用需同时满足：
1. `report.enabled=true`；
2. `departments.yaml` 里至少有一个部门配了邮箱；
否则启动时会打一条明确日志说明为什么没生效——避免"以为发了其实没发"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..config import Settings
from ..departments import DepartmentDirectory
from ..logging_setup import get_logger
from .mailer import deliver_weekly

logger = get_logger("insights.scheduler")

JOB_ID = "dept-weekly-report"
_VALID_WEEKDAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}


class WeeklyReportScheduler:
    """每周固定时刻把各部门周报投递出去。"""

    def __init__(self, settings: Settings, *, scheduler: Any | None = None) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._started = False
        self._last: dict[str, Any] | None = None
        self._runs = 0
        self._failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self._settings.report.enabled)

    @property
    def running(self) -> bool:
        return self._started and self._scheduler is not None and self._scheduler.running

    @property
    def weekday(self) -> str:
        value = (self._settings.report.weekday or "mon").strip().lower()[:3]
        if value not in _VALID_WEEKDAYS:
            logger.warning("report.weekday=%r 非法，按 mon 处理", self._settings.report.weekday)
            return "mon"
        return value

    def start(self) -> bool:
        if not self.enabled:
            logger.info("report.enabled=false，不启动周报定时投递")
            return False
        directory = DepartmentDirectory.from_settings(self._settings)
        if not any(department.email for department in directory.departments):
            logger.warning(
                "report.enabled=true 但 departments.yaml 里没有任何部门配邮箱，周报不会发出"
            )
            return False
        if self._started:
            return True

        if self._scheduler is None:
            from apscheduler.schedulers.background import BackgroundScheduler

            self._scheduler = BackgroundScheduler()

        config = self._settings.report
        self._scheduler.add_job(
            self._tick,
            trigger="cron",
            day_of_week=self.weekday,
            hour=int(config.hour),
            minute=int(config.minute),
            id=JOB_ID,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )
        self._scheduler.start()
        self._started = True
        logger.info(
            "周报定时投递已启动：每周%s %02d:%02d（%s）",
            self.weekday,
            int(config.hour),
            int(config.minute),
            "仅落盘预览" if config.dry_run else "真实发送",
        )
        return True

    def shutdown(self) -> None:
        if self._scheduler is not None and self._started:
            try:
                self._scheduler.remove_job(JOB_ID)
            except Exception:  # noqa: BLE001
                pass
            self._scheduler.shutdown(wait=False)
            logger.info("周报定时投递已停止")
        self._started = False

    # ---- 执行 ----
    def run_once(self, *, dry_run: bool | None = None) -> dict[str, Any] | None:
        """立即执行一次（看板的手动触发、测试都走这里）。"""
        try:
            result = deliver_weekly(self._settings, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - 定时线程不能崩
            self._failures += 1
            logger.exception("周报投递失败")
            return {"error": f"{type(exc).__name__}: {exc}"}
        self._runs += 1
        self._last = result
        return result

    def _tick(self) -> None:
        logger.info("周报定时任务触发")
        self.run_once()

    # ---- 状态 ----
    @property
    def next_run_time(self) -> datetime | None:
        if not self.running or self._scheduler is None:
            return None
        job = self._scheduler.get_job(JOB_ID)
        return getattr(job, "next_run_time", None) if job else None

    def status(self) -> dict[str, Any]:
        moment = self.next_run_time
        config = self._settings.report
        return {
            "enabled": self.enabled,
            "running": self.running,
            "dry_run": config.dry_run,
            "weekday": self.weekday,
            "time": f"{int(config.hour):02d}:{int(config.minute):02d}",
            "next_run_at": moment.isoformat() if moment else None,
            "runs": self._runs,
            "failures": self._failures,
            "last": self._last,
        }


__all__ = ["JOB_ID", "WeeklyReportScheduler"]
