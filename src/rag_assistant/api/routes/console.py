"""运维台接口：此刻在线多少、答得准不准、系统健不健康、最近出了什么错。

为什么单独一个路由文件：
- `dept.py` 是"业务看板"，口径是"学生问了什么、该补什么料"；
- 这里是"值班台"，口径是"服务现在怎么样"，用户是维护人员而不是部门负责人。
  两者的数据源不同（一个读日志，一个读进程内存 + 索引状态），混在一起会越来越乱。

鉴权：`/console` 前缀由 `AuthMiddleware` 限定为 dept / admin 角色，具体见
`auth.py` 的 `DEPT_PREFIXES` 与 `DEPT_EXACT`。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from ... import __version__
from ...config import Settings
from ...insights import load_records, summarize
from ...llm import llm_note
from ...insights.ops import (
    dir_size,
    human_size,
    latest_eval_summary,
    quality_report,
    recent_events,
)
from ...kb import KnowledgeBase
from ...logging_setup import get_logger
from ..deps import get_jobs, get_kb
from ..jobs import SyncJobManager
from .dept import _brand, _kick_sync, get_directory, get_settings_dep

router = APIRouter(tags=["console"])
logger = get_logger("api.console")

#: 每个接口都扫日志的话，5 秒一次的轮询太浪费；这里做个很短的缓存
_CACHE_TTL = 8.0
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, factory):
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < _CACHE_TTL:
        return hit[1]
    value = factory()
    _cache[key] = (now, value)
    return value


def _live(request: Request):
    service = request.app.state.chat
    tracker = getattr(service, "live", None)
    if tracker is None:  # pragma: no cover - 注入的服务没带计数器
        return {"online": 0, "inflight": 0, "single_process": True}
    return tracker.snapshot()


def _series(records: list[dict[str, Any]], *, minutes: int = 60) -> list[dict[str, int]]:
    """最近 N 分钟每分钟的提问数与拒答数（给页面画迷你趋势）。"""
    now = datetime.now()
    buckets: dict[str, dict[str, int]] = {}
    for offset in range(minutes - 1, -1, -1):
        stamp = now.replace(second=0, microsecond=0).timestamp() - offset * 60
        key = datetime.fromtimestamp(stamp).strftime("%H:%M")
        buckets[key] = {"minute": key, "total": 0, "refused": 0}

    for record in records:
        raw = record.get("ts") or record.get("_ts")
        if not isinstance(raw, str):
            continue
        try:
            moment = datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            continue
        if (now - moment).total_seconds() > minutes * 60:
            continue
        key = moment.strftime("%H:%M")
        bucket = buckets.get(key)
        if bucket is None:
            continue
        bucket["total"] += 1
        if record.get("refused"):
            bucket["refused"] += 1
    return list(buckets.values())


@router.get("/console/live", summary="运维台：实时在线与最近一分钟的负载")
def console_live(request: Request, settings: Settings = Depends(get_settings_dep)) -> dict:
    snapshot = _live(request)

    def _build() -> list[dict[str, int]]:
        records = load_records(
            settings.log_dir, qa_log=settings.logging.qa_log, days=1, now=datetime.now()
        )
        return _series(records)

    snapshot = dict(snapshot)
    snapshot["series"] = _cached("series", _build)
    snapshot["brand"] = _brand(settings)
    snapshot["server_time"] = datetime.now().isoformat(timespec="seconds")
    return snapshot


@router.get("/console/quality", summary="运维台：回答质量（应答率/引用覆盖/拒答原因/耗时）")
def console_quality(
    days: int = Query(7, ge=1, le=90, description="统计窗口（天）"),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    moment = datetime.now()

    def _build() -> dict:
        records = load_records(
            settings.log_dir, qa_log=settings.logging.qa_log, days=days, now=moment
        )
        report = quality_report(records, days=days, now=moment)
        report["eval"] = latest_eval_summary(settings.reports_dir)
        report["notes"] = {
            "answer_rate": "有作答的比例；不等于答得对，只说明没有触发拒答。",
            "citation_rate": "作答里带引用的比例，用来盯「每句有出处」这条底线。",
            "claims": "逐句校验的支持率，只有开启逐句校验后产生的问答才有样本。",
            "eval": "来自 reports/ 下最近一份离线评测报告，与线上日志是两回事。",
        }
        return report

    return _cached(f"quality:{days}", _build)


@router.get("/console/system", summary="运维台：索引、模型、磁盘与后台任务状态")
def console_system(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    kb: KnowledgeBase = Depends(get_kb),
    jobs: SyncJobManager = Depends(get_jobs),
) -> dict:
    status = kb.status()
    service = request.app.state.chat
    scheduler = getattr(request.app.state, "scheduler", None)
    watcher = getattr(request.app.state, "watcher", None)
    weekly = getattr(request.app.state, "weekly", None)
    started_at = getattr(request.app.state, "started_at", None)

    index_dir = settings.index_dir
    logs_dir = settings.log_dir
    manifest = status.get("manifest") if isinstance(status.get("manifest"), dict) else {}
    return {
        "version": __version__,
        "started_at": started_at,
        "uptime_seconds": int(time.time() - started_at) if started_at else None,
        "brand": _brand(settings),
        "index": {
            "documents": status.get("documents", 0),
            "chunks": status.get("manifest_chunks", 0),
            "vector_chunks": status.get("vector_chunks", 0),
            "bm25_chunks": status.get("bm25_chunks", 0),
            "collection": status.get("collection", ""),
            "last_sync": status.get("last_sync"),
            "size": human_size(dir_size(index_dir)),
            "failed_docs": manifest.get("failed") or status.get("failed") or [],
        },
        "models": {
            "llm": getattr(service.llm, "model", "unknown"),
            "llm_mock": bool(getattr(service.llm, "is_mock", False)),
            "llm_note": llm_note(service.llm),
            "embedder": getattr(service.kb.embedder, "name", "unknown"),
            "embedder_device": getattr(service.kb.embedder, "device", "unknown"),
            "embedder_mock": bool(getattr(service.kb.embedder, "is_mock", True)),
            "reranker": getattr(getattr(service.kb, "reranker", None), "name", "off"),
            "reranker_device": getattr(getattr(service.kb, "reranker", None), "device", "-"),
        },
        "jobs": {
            "current": jobs.current.to_dict() if jobs.current else None,
            "recent": [job.to_dict() for job in jobs.history[-5:]][::-1],
        },
        "background": {
            "sync_scheduler": bool(getattr(scheduler, "running", False)),
            "sync_interval_seconds": settings.sync.interval_seconds,
            "watcher": bool(getattr(watcher, "active", False)),
            "weekly_report": weekly.status() if weekly is not None else {"enabled": False},
        },
        "disk": {
            "logs": human_size(dir_size(logs_dir)),
            "qa_log": human_size(dir_size(logs_dir / settings.logging.qa_log)),
            "reports": human_size(dir_size(settings.reports_dir)),
        },
        "paths": {
            "raw_dir": str(settings.raw_dir),
            "index_dir": str(index_dir),
            "log_dir": str(logs_dir),
        },
    }


@router.get("/console/events", summary="运维台：最近的应用错误/告警")
def console_events(
    limit: int = Query(30, ge=1, le=200),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    path: Path = settings.log_dir / "app.log"
    events = recent_events(path, limit=limit)
    return {
        "file": str(path),
        "exists": path.exists(),
        "errors": sum(1 for item in events if item["level"] in {"ERROR", "CRITICAL"}),
        "warnings": sum(1 for item in events if item["level"] == "WARNING"),
        "events": events,
    }


@router.get("/console/top", summary="运维台：高频问题与被拒问题（与部门看板同源）")
def console_top(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(8, ge=1, le=30),
    settings: Settings = Depends(get_settings_dep),
    directory=Depends(get_directory),
) -> dict:
    moment = datetime.now()

    def _build() -> dict:
        records = load_records(
            settings.log_dir, qa_log=settings.logging.qa_log, days=days, now=moment
        )
        payload = summarize(
            records, directory=directory, days=days, now=moment, top_n=limit
        )
        return {
            "window_days": days,
            "total": payload.get("total", 0),
            "top_questions": payload.get("top_questions", [])[:limit],
            "refused_questions": payload.get("refused_questions", [])[:limit],
            "departments": payload.get("departments", [])[:limit],
            "unmatched": payload.get("unmatched", 0),
        }

    return _cached(f"top:{days}:{limit}", _build)


@router.post("/console/sync", summary="运维台：立即同步（可选重建）")
def console_sync(
    rebuild: bool = Query(False, description="true 时整库重建"),
    jobs: SyncJobManager = Depends(get_jobs),
) -> dict:
    if rebuild:
        from ...exceptions import IndexBusyError

        try:
            job = jobs.trigger(rebuild=True)
            return {"job_id": job.job_id, "status": job.status, "rebuild": True}
        except IndexBusyError as exc:
            return {"job_id": None, "status": "busy", "detail": str(exc)}
    return _kick_sync(jobs)
