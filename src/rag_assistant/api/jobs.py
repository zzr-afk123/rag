"""后台同步任务管理。

为什么不用 FastAPI 的 ``BackgroundTasks``：它依附于单次请求的生命周期，
没有"任务 ID / 状态查询 / 拒绝重复触发"这些能力，而 ``POST /kb/sync`` 需要它们
（需求 5：手动触发同步 + 状态可查）。

并发语义：
- **同一时刻只允许一个同步任务**（``IndexBusyError`` → HTTP 409）。
  这不是偷懒：向量库、BM25、manifest 三者的一致性靠"写入串行"保证，
  两个同步同时跑会互相覆盖。
- 阶段 E 会把这个进程内锁换成读写锁（读=查询，写=同步），并加定时触发。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..exceptions import IndexBusyError
from ..kb import KnowledgeBase, SyncReport
from ..logging_setup import get_logger

logger = get_logger("api.jobs")


@dataclass
class SyncJob:
    """一次同步任务的状态。"""

    job_id: str
    rebuild: bool = False
    status: str = "pending"  # pending | running | success | failed
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    report: dict[str, Any] | None = None
    error: str | None = None

    @property
    def running(self) -> bool:
        return self.status in {"pending", "running"}

    @property
    def duration_ms(self) -> float | None:
        if self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at) * 1000, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "rebuild": self.rebuild,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "report": self.report,
            "error": self.error,
        }


class SyncJobManager:
    """单飞（single-flight）同步任务管理。"""

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb
        self._lock = threading.Lock()
        self._current: SyncJob | None = None
        self._history: list[SyncJob] = []

    # ---- 状态 ----
    @property
    def current(self) -> SyncJob | None:
        return self._current

    @property
    def is_running(self) -> bool:
        job = self._current
        return bool(job and job.running)

    @property
    def last(self) -> SyncJob | None:
        return self._history[-1] if self._history else None

    @property
    def history(self) -> list[SyncJob]:
        """最近的任务（按触发顺序，最早的在前）。运维台只读展示用。"""
        with self._lock:
            return list(self._history)

    @property
    def last_error(self) -> str | None:
        last = self.last
        return last.error if last else None

    @property
    def last_report(self) -> dict[str, Any] | None:
        last = self.last
        return last.report if last else None

    def get(self, job_id: str) -> SyncJob | None:
        """按 id 查任务（上传后前端轮询进度用）。只保留本进程内的历史。"""
        with self._lock:
            for job in self._history:
                if job.job_id == job_id:
                    return job
        return None

    # ---- 触发 ----
    def trigger(self, *, rebuild: bool = False) -> SyncJob:
        """后台启动同步，立即返回任务。已有任务在跑时抛 :class:`IndexBusyError`。"""
        with self._lock:
            if self._current is not None and self._current.running:
                raise IndexBusyError(
                    f"已有同步任务在执行（job_id={self._current.job_id}），请稍后再试"
                )
            job = SyncJob(job_id=uuid.uuid4().hex[:12], rebuild=rebuild)
            self._current = job
            self._history.append(job)

        thread = threading.Thread(target=self._run, args=(job,), name=f"kb-sync-{job.job_id}", daemon=True)
        thread.start()
        logger.info("已提交同步任务 %s（rebuild=%s）", job.job_id, rebuild)
        return job

    def run_inline(self, *, rebuild: bool = False) -> SyncJob:
        """同步执行（``wait=true`` 用）。同样受单飞约束。"""
        with self._lock:
            if self._current is not None and self._current.running:
                raise IndexBusyError(
                    f"已有同步任务在执行（job_id={self._current.job_id}），请稍后再试"
                )
            job = SyncJob(job_id=uuid.uuid4().hex[:12], rebuild=rebuild)
            self._current = job
            self._history.append(job)
        self._run(job)
        return job

    # ---- 执行 ----
    def _run(self, job: SyncJob) -> None:
        job.status = "running"
        try:
            report: SyncReport = self._kb.sync(rebuild=job.rebuild)
            job.report = report.to_dict()
            job.status = "failed" if report.failed else "success"
            if report.failed:
                job.error = f"{len(report.failed)} 个文件同步失败"
        except Exception as exc:  # noqa: BLE001 - 任务失败不能把线程打崩
            logger.exception("同步任务失败：%s", job.job_id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()
            logger.info(
                "同步任务结束 %s：status=%s 耗时=%sms",
                job.job_id,
                job.status,
                job.duration_ms,
            )

    # ---- 等待（测试与脚本用） ----
    def wait(self, timeout: float = 120.0, *, poll: float = 0.02) -> SyncJob | None:
        """阻塞等待当前任务结束。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self._current
            if job is None or not job.running:
                return job
            time.sleep(poll)
        return self._current
