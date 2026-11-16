"""部门看板接口：近 N 天的问题分布、周报预览与投递，以及拖拽上传入库。

数据全部来自 `logs/qa.jsonl`（经由 `insights.analytics`），**不暴露问答正文**，
只暴露"问题文本 + 是否作答 + 部门归属"——这已经够部门判断"该补哪份文件"，
而回答正文里可能含个人信息与制度原文细节，不往外发。

**上传入库**（`POST /dept/upload`）：把文件落到 `data/raw/uploads/`，再触发一次
增量同步，把"发现缺料 → 补文件 → 立即可检索"接成闭环。上传刻意不用
FastAPI 的 `UploadFile`：那要求 `python-multipart`，为一次上传引入新依赖不划算。
请求体就是文件字节，文件名走 `?name=`；落盘用"临时文件 + replace"，
避免定时同步扫到写了一半的文件。

鉴权：`server` 层的中间件已经对 `/dept/` 前缀做了令牌校验（`dept_console.token`），
这里不再重复实现。页面 `/dept` 本身不校验，否则用户没法先打开页面填令牌。
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from ...config import Settings
from ...departments import DepartmentDirectory
from ...exceptions import IndexBusyError, ManifestError
from ...ingestion.manifest import load_manifest
from ...insights import (
    collect_gaps,
    dept_slice,
    deliver_weekly,
    gaps_markdown,
    load_records,
    summarize,
    weekly_markdown,
    weekly_workbook,
)
from ...kb import KnowledgeBase
from ...logging_setup import get_logger
from ..deps import get_jobs, get_kb
from ..jobs import SyncJobManager

router = APIRouter(tags=["dept"])
logger = get_logger("api.dept")

_MAX_WINDOW = 180

#: 上传文件的落盘子目录（在 `sync.watch_dir` 之下，会被同步扫描到）
_UPLOAD_DIR = "uploads"
#: 文件名主干的最大字符数（给扩展名和 Windows 路径长度留余量）
_MAX_NAME_CHARS = 100
#: Windows 文件名非法字符（含控制字符）；斜杠也一并替换，避免路径穿越
_UNSAFE_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


# --------------------------------------------------------------------------
# 上传：文件名收敛 / 体积限制 / 原子落盘
# --------------------------------------------------------------------------
def _upload_root(settings: Settings) -> Path:
    """上传目录：`<raw_dir>/uploads`（不存在时按需创建）。"""
    return settings.raw_dir / _UPLOAD_DIR


def _allowed_suffixes(settings: Settings) -> set[str]:
    """从 `sync.include_patterns` 推导允许上传的扩展名（配置是唯一事实源）。"""
    suffixes = {
        pattern[1:].lower()
        for pattern in settings.sync.include_patterns
        if pattern.startswith("*.")
    }
    return suffixes or {".pdf", ".docx", ".md", ".txt", ".xlsx"}


def _sanitize_name(raw: str, allowed: set[str]) -> str:
    """把客户端文件名收敛成安全的、只落在 `uploads/` 下的文件名。

    只取最后一段（挡掉 `../../etc/passwd`）、去掉 Windows 非法字符与首尾点，
    并拒绝不在白名单里的扩展名。
    """
    name = str(raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE_NAME.sub("_", name).strip().strip(".")
    if not name:
        raise HTTPException(status_code=400, detail="文件名不合法（为空或只含分隔符）")

    stem, dot, suffix = name.rpartition(".")
    suffix = f".{suffix.lower()}" if dot else ""
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型：{suffix or '（无扩展名）'}；仅支持 {' / '.join(sorted(allowed))}",
        )
    stem = stem.strip().strip(".")[:_MAX_NAME_CHARS] or "未命名文件"
    if stem.lower() in _RESERVED_NAMES:
        stem = f"_{stem}"
    return f"{stem}{suffix}"


def _too_large_detail(max_bytes: int) -> str:
    return f"文件太大：单个文件上限 {max_bytes // (1024 * 1024)} MB"


async def _read_upload_body(request: Request, max_bytes: int) -> bytes:
    """流式读取请求体，超限立刻中断（不把超大文件整个读进内存）。"""
    declared = (request.headers.get("content-length") or "").strip()
    if declared.isdigit() and int(declared) > max_bytes:
        raise HTTPException(status_code=413, detail=_too_large_detail(max_bytes))

    buffer = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(buffer) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=_too_large_detail(max_bytes))
        buffer.extend(chunk)
    return bytes(buffer)


def _store_upload(data: bytes, target: Path, *, overwrite: bool) -> tuple[Path, bool]:
    """原子落盘：先写隐藏临时文件再 replace，避免同步扫到半截文件。

    返回 `(最终路径, 是否覆盖了已有文件)`；同名且未要求覆盖时抛 409。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".upload-", suffix=".part", dir=str(target.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        existed = target.exists()
        if existed and not overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"已存在同名文件：{target.name}；需要替换请勾选「覆盖」后重试",
            )
        os.replace(tmp, target)
        return target, existed
    finally:
        tmp.unlink(missing_ok=True)


def _kick_sync(jobs: SyncJobManager) -> dict[str, Any]:
    """触发一次增量同步；已有任务在跑时返回当前任务而不是报错（文件已安全落盘）。"""
    try:
        job = jobs.trigger(rebuild=False)
        return {"job_id": job.job_id, "status": job.status}
    except IndexBusyError:
        current = jobs.current
        return {
            "job_id": current.job_id if current else None,
            "status": "busy",
            "detail": "已有同步任务在执行；新文件会在这次任务之后补同步（默认 60s 内），也可点「立即同步」。",
        }


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_directory(request: Request) -> DepartmentDirectory:
    return request.app.state.departments


@router.get("/dept/summary", summary="部门看板：近 N 天统计")
def summary(
    request: Request,
    days: int = Query(0, ge=0, le=_MAX_WINDOW, description="统计窗口天数，0 表示用配置值"),
    settings: Settings = Depends(get_settings_dep),
    directory: DepartmentDirectory = Depends(get_directory),
) -> dict:
    window = days or settings.dept_console.window_days
    moment = datetime.now()
    records = load_records(
        settings.log_dir, qa_log=settings.logging.qa_log, days=window, now=moment
    )
    payload = summarize(
        records,
        directory=directory,
        days=window,
        now=moment,
        top_n=settings.dept_console.top_n,
    )
    weekly = getattr(request.app.state, "weekly", None)
    return {
        **payload,
        "brand": _brand(settings),
        "departments_catalog": directory.catalog(),
        "report": weekly.status() if weekly is not None else {"enabled": False},
        "upload": {
            "max_mb": settings.dept_console.upload_max_mb,
            "types": sorted(_allowed_suffixes(settings)),
        },
        "notes": {
            "counts": "统计窗口内的问答次数（含拒答）；同一问题重复提问按次计。",
            "departments": "部门归属按关键词命中判定，一条提问可能同时命中多个部门。",
            "privacy": "只统计问题与是否作答，不含回答正文与个人信息。",
        },
    }


def _attachment_headers(raw_name: str, fallback: str = "report.md") -> dict[str, str]:
    """构造附件下载头：中文名走 RFC 5987，ASCII 兜底名保证老客户端也能存盘。

    ``fallback`` 由调用方给出（如 ``jwc-30d.md``）。不要从中文名派生子串——
    "郑大-补料清单-30d.md" 去掉中文后只剩 "_-_-30d.md"，存盘名很难看。
    """
    # HTTP 头只能是 latin-1；中文名必须走 filename*，否则 Starlette 编码响应头时
    # 抛 UnicodeEncodeError，浏览器收到 500（这就是"下载周报点了没反应"的根因）。
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", fallback).strip("-.") or "report.md"
    return {
        "Content-Disposition": (
            f'attachment; filename="{safe}"; filename*=UTF-8\'\'{quote(raw_name)}'
        )
    }


@router.get("/dept/gaps", summary="知识库缺口清单（按部门分组，供补料闭环）")
def gaps(
    days: int = Query(0, ge=0, le=_MAX_WINDOW, description="统计窗口天数，0 表示用看板配置值"),
    settings: Settings = Depends(get_settings_dep),
    directory: DepartmentDirectory = Depends(get_directory),
) -> dict:
    """把"没答上来的问题"按部门聚合成补料清单。

    与看板读同一份 `qa.jsonl`、同一套部门归属规则，所以清单与页面数字一致。
    """
    window = days or settings.dept_console.window_days
    moment = datetime.now()
    records = load_records(
        settings.log_dir, qa_log=settings.logging.qa_log, days=window, now=moment
    )
    payload = collect_gaps(records, directory=directory, days=window, now=moment)
    payload["brand"] = _brand(settings)
    return payload


@router.get(
    "/dept/gaps/download",
    summary="下载《制度文件补充清单》（Markdown）",
    response_class=PlainTextResponse,
)
def gaps_download(
    days: int = Query(0, ge=0, le=_MAX_WINDOW),
    settings: Settings = Depends(get_settings_dep),
    directory: DepartmentDirectory = Depends(get_directory),
):
    window = days or settings.dept_console.window_days
    moment = datetime.now()
    records = load_records(
        settings.log_dir, qa_log=settings.logging.qa_log, days=window, now=moment
    )
    payload = collect_gaps(records, directory=directory, days=window, now=moment)
    markdown = gaps_markdown(
        payload,
        school_name=settings.branding.school_name,
        site_title=settings.branding.site_title,
        example_data=True,
    )
    raw_name = f"{settings.branding.short_name or 'kb'}-补料清单-{window}d.md"
    return PlainTextResponse(
        markdown,
        media_type="text/markdown; charset=utf-8",
        headers=_attachment_headers(raw_name, fallback=f"gap-report-{window}d.md"),
    )


@router.get("/dept/status", summary="部门看板：周报任务状态")
def status(request: Request) -> dict:
    weekly = getattr(request.app.state, "weekly", None)
    return {
        "report": weekly.status() if weekly is not None else {"enabled": False},
        "log_dir": str(request.app.state.settings.log_dir),
    }


@router.get(
    "/dept/report",
    summary="部门周报（默认 Markdown 预览，download=true 时导出 Excel）",
    response_class=PlainTextResponse,
)
def report(
    dept: str = Query("", description="部门 id；留空表示全校汇总"),
    days: int = Query(0, ge=0, le=_MAX_WINDOW, description="统计窗口天数，0 表示用配置值"),
    download: bool = Query(False, description="true 时以 Excel 附件下载"),
    settings: Settings = Depends(get_settings_dep),
    directory: DepartmentDirectory = Depends(get_directory),
):
    window = days or settings.report.days
    moment = datetime.now()
    records = load_records(
        settings.log_dir, qa_log=settings.logging.qa_log, days=window, now=moment
    )
    payload = summarize(
        records, directory=directory, days=window, now=moment, top_n=settings.dept_console.top_n
    )
    department = directory.by_id(dept) if dept else None
    if dept and department is None:
        return PlainTextResponse(f"未知部门：{dept}", status_code=404)
    if download:
        # 看板「导出周报」给的是 Excel：值班主任要拿它贴月报、排序筛选，
        # Markdown 只能看不能算。纯文本预览仍走下面的 Markdown 分支。
        data = weekly_workbook(
            payload,
            department=department,
            school_name=settings.branding.school_name,
            site_title=settings.branding.site_title,
            example_data=True,
            dept_names={row.id: row.name for row in directory.departments},
        )
        slug = department.id if department else "all"
        raw_name = f"{settings.branding.short_name or 'dept'}-{slug}-{window}d.xlsx"
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=_attachment_headers(raw_name, fallback=f"{slug}-{window}d.xlsx"),
        )
    markdown = weekly_markdown(
        payload,
        department=department,
        school_name=settings.branding.school_name,
        site_title=settings.branding.site_title,
        example_data=True,
    )
    return PlainTextResponse(markdown, media_type="text/markdown; charset=utf-8")


@router.post("/dept/report/send", summary="立即投递周报（默认只落盘，不真发）")
def send(
    days: int = Query(0, ge=0, le=_MAX_WINDOW),
    dept: str = Query("", description="只发某个部门（调试用）"),
    dry_run: bool | None = Query(None, description="覆盖配置里的 report.dry_run"),
    settings: Settings = Depends(get_settings_dep),
    directory: DepartmentDirectory = Depends(get_directory),
) -> dict:
    window = days or settings.report.days
    result = deliver_weekly(
        settings,
        directory=directory,
        days=window,
        dry_run=dry_run,
        only=dept or None,
    )
    logger.info(
        "手动投递周报：窗口 %d 天，部门 %s，dry_run=%s",
        window,
        dept or "全部",
        result.get("dry_run"),
    )
    return result


@router.get("/dept/department/{dept_id}", summary="单个部门的明细（看板下钻）")
def department_detail(
    dept_id: str,
    days: int = Query(0, ge=0, le=_MAX_WINDOW),
    settings: Settings = Depends(get_settings_dep),
    directory: DepartmentDirectory = Depends(get_directory),
) -> dict:
    department = directory.by_id(dept_id)
    if department is None:
        return {"error": f"未知部门：{dept_id}"}
    window = days or settings.dept_console.window_days
    moment = datetime.now()
    records = load_records(
        settings.log_dir, qa_log=settings.logging.qa_log, days=window, now=moment
    )
    payload = summarize(
        records, directory=directory, days=window, now=moment, top_n=settings.dept_console.top_n
    )
    return {
        "department": department.to_dict(),
        "window_days": window,
        "summary": dept_slice(payload, dept_id),
        "total": payload.get("total", 0),
        "generated_at": payload.get("generated_at", ""),
    }


# --------------------------------------------------------------------------
# 上传入库
# --------------------------------------------------------------------------
@router.post(
    "/dept/upload",
    summary="上传制度文件（请求体为文件字节，文件名走 ?name=）",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        }
    },
)
async def upload(
    request: Request,
    name: str = Query(..., min_length=1, max_length=260, description="原始文件名（含扩展名）"),
    overwrite: bool = Query(False, description="同名文件是否覆盖"),
    settings: Settings = Depends(get_settings_dep),
    jobs: SyncJobManager = Depends(get_jobs),
) -> dict:
    """把文件写进 `data/raw/uploads/`，随后触发增量同步。

    前端用 ``fetch(url, {method: "POST", body: file})``，请求体就是文件原始字节；
    文件名放查询参数（不引入 multipart 依赖，也不依赖自定义请求头）。
    """
    user = getattr(request.state, "user", None)
    uploader = str(
        getattr(user, "id", "") or (request.client.host if request.client else "unknown")
    )
    limiter = getattr(request.app.state, "upload_limiter", None)
    if limiter is not None and limiter.limit > 0:
        allowed, retry_after = limiter.check(f"upload:{uploader}")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"上传过于频繁（每分钟 {limiter.limit} 个文件），请 {retry_after}s 后再试",
                headers={"Retry-After": str(retry_after)},
            )

    filename = _sanitize_name(name, _allowed_suffixes(settings))
    max_bytes = max(1, settings.dept_console.upload_max_mb) * 1024 * 1024
    data = await _read_upload_body(request, max_bytes)
    if not data:
        raise HTTPException(status_code=400, detail="文件内容为空")

    target, overwritten = await run_in_threadpool(
        _store_upload, data, _upload_root(settings) / filename, overwrite=overwrite
    )
    sync = _kick_sync(jobs)
    logger.info(
        "上传入库：%s 写入 %s（%d 字节，覆盖=%s），同步任务 %s",
        uploader,
        target.name,
        len(data),
        overwritten,
        sync.get("job_id"),
    )
    return {
        "saved": True,
        "name": target.name,
        "rel_path": f"{_UPLOAD_DIR}/{target.name}",
        "size": len(data),
        "overwritten": overwritten,
        "uploaded_at": time.time(),
        "sync": sync,
    }


@router.post("/dept/sync", summary="触发一次增量同步（上传后补同步用）")
def trigger_sync(jobs: SyncJobManager = Depends(get_jobs)) -> dict:
    """与 `/kb/sync` 等价，但走部门看板的鉴权与单飞语义；忙时返回当前任务而不是 409。"""
    return _kick_sync(jobs)


@router.get("/dept/sync/{job_id}", summary="查询同步任务状态与入库结果")
def sync_job(
    job_id: str,
    jobs: SyncJobManager = Depends(get_jobs),
    kb: KnowledgeBase = Depends(get_kb),
) -> dict:
    """轮询用：任务结束后附带知识库规模，前端据此显示「已入库 N 篇 / M 片段」。"""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"没有这个同步任务：{job_id}")
    payload: dict[str, Any] = {"job": job.to_dict()}
    if not job.running:
        status = kb.status()
        payload["kb"] = {
            "documents": status.get("documents", 0),
            "chunks": status.get("manifest_chunks", 0),
            "vector_chunks": status.get("vector_chunks", 0),
            "last_sync": status.get("last_sync"),
        }
    return payload


@router.get("/dept/uploads", summary="最近上传的文件与入库状态")
def recent_uploads(
    limit: int = Query(8, ge=1, le=50),
    settings: Settings = Depends(get_settings_dep),
) -> dict:
    """按修改时间倒序列出 `uploads/` 里的文件，并用 manifest 标注是否已入库。"""
    root = _upload_root(settings)
    if not root.exists():
        return {"files": [], "upload_dir": str(root)}

    files = [path for path in root.glob("*") if path.is_file() and not path.name.startswith(".")]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    try:
        manifest = load_manifest(settings.manifest_path)
    except ManifestError:
        manifest = None
    indexed = (
        {entry.rel_path: entry for entry in manifest.docs.values()} if manifest is not None else {}
    )

    payload = []
    for path in files[:limit]:
        entry = indexed.get(f"{_UPLOAD_DIR}/{path.name}")
        stat = path.stat()
        payload.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "indexed": entry is not None,
                "chunks": entry.chunks if entry else 0,
                "indexed_at": entry.indexed_at if entry else None,
            }
        )
    return {"files": payload, "upload_dir": str(root)}


def _brand(settings: Settings) -> dict[str, object]:
    branding = settings.branding
    return {
        "school_name": branding.school_name,
        "short_name": branding.short_name,
        "motto": branding.motto,
        "site_title": branding.site_title,
        "primary_color": branding.primary_color,
        "accent_color": branding.accent_color,
        "footer_note": branding.footer_note,
    }
