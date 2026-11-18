"""健康检查。

刻意返回"当前 LLM/Embedding 是否 mock"：部署时最常见的误配是
"明明以为接了 DeepSeek，其实没配 Key 一直跑 mock"，这里一眼就能看出来。

另外返回 ``lan_urls``：服务器模式（``server.host=0.0.0.0``）下给出局域网地址，
手机打开页面时会直接显示"手机访问地址"，不用用户自己去查 ipconfig。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...chat_service import ChatService
from ...config import Settings
from ...llm import llm_note
from ..deps import get_chat, get_settings_dep
from ..netinfo import build_urls
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health(
    service: ChatService = Depends(get_chat),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    host = settings.server.host
    port = settings.server.port
    describe = getattr(service.llm, "describe", None)
    route_info = describe() if callable(describe) else {}
    live = getattr(service, "live", None)
    snapshot = live.snapshot() if live is not None else {}
    return HealthResponse(
        status="ok",
        llm=getattr(service.llm, "model", "unknown"),
        llm_mock=service.llm.is_mock,
        embedder=service.kb.embedder.name,
        embedder_mock=service.kb.embedder.is_mock,
        chunks=service.kb.vector_store.count(),
        score_threshold=settings.retrieval.score_threshold,
        lan_urls=build_urls(host, port),
        host=host,
        port=port,
        llm_fallback=str(route_info.get("fallback", "")),
        llm_active=str(route_info.get("active", "")),
        llm_note=llm_note(service.llm),
        public_url=settings.server.public_url,
        online=int(snapshot.get("online", 0)),
        questions_per_minute=int(snapshot.get("questions_per_minute", 0)),
    )
