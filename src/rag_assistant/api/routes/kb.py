"""知识库路由：状态查询 + 手动同步。

同步默认**异步**（202 + job_id）：制度库可能上千文件，同步耗时几十秒，
同步接口容易在网关/客户端超时。``wait=true`` 保留给调试与 CI。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from ...kb import KnowledgeBase
from ..deps import get_jobs, get_kb
from ..jobs import SyncJobManager
from ..schemas import KBStatus, SyncAccepted, SyncCompleted, SyncRequest

router = APIRouter(tags=["kb"])


@router.get("/kb/chunks/{chunk_id}", summary="取回某个 chunk 的原文（引用回查）")
def kb_chunk(chunk_id: str, kb: KnowledgeBase = Depends(get_kb)) -> dict:
    """返回被引用片段的完整正文与章节信息。

    答案里的 ``[1]`` 只带 120 字摘录；点开要看的是"原文那一段"，
    所以前端按 ``citation.chunk_id`` 回查这里，并把引文高亮出来。
    """
    chunks = kb.get_chunks([chunk_id])
    if not chunks:
        return JSONResponse(
            status_code=404,
            content={"detail": f"没有这个片段：{chunk_id}（索引可能已重建）"},
        )
    return chunks[0]


@router.get("/kb/status", response_model=KBStatus, summary="知识库索引状态")
def kb_status(
    request: Request,
    kb: KnowledgeBase = Depends(get_kb),
    jobs: SyncJobManager = Depends(get_jobs),
) -> KBStatus:
    """返回索引规模、最后同步时间、当前是否正在同步、上次错误。"""
    status = kb.status()
    current = jobs.current
    scheduler = getattr(request.app.state, "scheduler", None)
    watcher = getattr(request.app.state, "watcher", None)
    moment = getattr(scheduler, "next_run_time", None)
    return KBStatus(
        documents=status.get("documents", 0),
        manifest_chunks=status.get("manifest_chunks", 0),
        vector_chunks=status.get("vector_chunks", 0),
        bm25_chunks=status.get("bm25_chunks", 0),
        last_sync=status.get("last_sync"),
        collection=status.get("collection", ""),
        embedder=status.get("embedder", ""),
        embedder_mock=bool(status.get("embedder_mock", False)),
        reranker=status.get("reranker"),
        score_threshold=float(status.get("score_threshold", 0.0)),
        syncing=jobs.is_running,
        last_job_id=current.job_id if current else None,
        last_error=jobs.last_error,
        last_report=jobs.last_report,
        scheduler_enabled=bool(getattr(scheduler, "enabled", False)),
        next_sync_at=moment.isoformat() if moment else None,
        watcher_active=bool(getattr(watcher, "active", False)),
        lock=status.get("lock"),
    )


@router.post(
    "/kb/sync",
    response_model=SyncAccepted,
    status_code=202,
    responses={200: {"model": SyncCompleted}, 409: {"description": "已有同步任务在执行"}},
    summary="手动触发同步（默认异步）",
)
def kb_sync(
    request: SyncRequest,
    jobs: SyncJobManager = Depends(get_jobs),
):
    """触发一次知识库同步。

    - 默认：后台执行，返回 202 与 ``job_id``，用 ``/kb/status`` 轮询 ``syncing``；
    - ``wait=true``：同步执行，返回 200 与完整报告（仅供调试/CI）。
    """
    if request.wait:
        job = jobs.run_inline(rebuild=request.rebuild)
        return JSONResponse(
            status_code=200,
            content=SyncCompleted(
                job_id=job.job_id,
                status=job.status,
                report=job.report,
                error=job.error,
            ).model_dump(mode="json"),
        )

    job = jobs.trigger(rebuild=request.rebuild)
    return JSONResponse(
        status_code=202,
        content=SyncAccepted(
            job_id=job.job_id,
            status=job.status,
            message="同步已在后台启动，请轮询 /kb/status 或使用该 job_id 查询",
        ).model_dump(mode="json"),
    )
