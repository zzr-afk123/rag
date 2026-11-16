"""问答路由：同步 + 流式（SSE）。

SSE 帧格式（前端按 ``event`` 分派）::

    event: meta
    data: {"type": "meta", "rewritten_query": "...", "retrieval": {...}}

    event: token
    data: {"type": "token", "text": "根据"}

    event: done
    data: {"type": "done", "answer": "...", "citations": [...]}

    data: [DONE]

为什么还要走一遍 ``event:`` 字段：浏览器 ``EventSource`` 只支持 GET，
本项目用 POST + fetch 流式读取，自己解析即可；但保留 ``event:`` 让
任何标准 SSE 客户端（curl / 第三方库）都能正确分派。
"""

from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...chat_service import ChatService
from ...logging_setup import get_logger
from ..deps import get_chat
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])
logger = get_logger("api.chat")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 关掉 nginx 的响应缓冲，否则流式会被攒成一整段再发
    "X-Accel-Buffering": "no",
}


def _sse_frame(event: dict) -> str:
    kind = str(event.get("type", "message"))
    payload = json.dumps(event, ensure_ascii=False)
    return f"event: {kind}\ndata: {payload}\n\n"


def _owner(request: Request) -> str | None:
    """取出当前登录身份，用于给会话 key 加前缀。"""
    user = getattr(request.state, "user", None)
    return getattr(user, "id", None) or None


def _threshold(request: Request, body: ChatRequest) -> float | None:
    """相关性阈值以**服务端配置**为准，客户端只能建议。

    ``threshold=0`` 等于把"证据不足就拒答"这道闸门整个关掉：问天气也能拿实验室条款硬答，
    注入者还能用它把闸门 1 旁路掉做对比测试。评测与运维确实要扫阈值，
    所以保留管理员覆盖；其余角色（含未开鉴权时的匿名请求）返回 ``None``，
    由 ``ChatService`` 回落 ``settings.retrieval.score_threshold``。
    """
    if body.threshold is None:
        return None
    user = getattr(request.state, "user", None)
    if getattr(user, "role", "") == "admin":
        return body.threshold
    logger.info("忽略客户端传入的阈值 %.2f（非管理员），改用 settings.retrieval.score_threshold", body.threshold)
    return None


@router.post("/chat", response_model=ChatResponse, summary="问答（同步）")
def chat(
    body: ChatRequest, request: Request, service: ChatService = Depends(get_chat)
) -> ChatResponse:
    """一次性返回完整回答与引用来源。"""
    reply = service.chat(
        body.session_id, body.question, threshold=_threshold(request, body), owner=_owner(request)
    )
    return ChatResponse.from_reply(reply)


@router.post("/chat/stream", summary="问答（SSE 流式）")
def chat_stream(
    body: ChatRequest, request: Request, service: ChatService = Depends(get_chat)
) -> StreamingResponse:
    """流式返回：``meta`` → 若干 ``token`` → ``done``（或 ``retract``）。"""
    owner = _owner(request)

    def generate() -> Iterator[str]:
        try:
            for event in service.chat_stream(
                body.session_id,
                body.question,
                threshold=_threshold(request, body),
                owner=owner,
            ):
                yield _sse_frame(event)
        except Exception as exc:  # noqa: BLE001 - 流已开始，只能以事件形式报错
            logger.exception("流式问答异常")
            # 只给错误类别，不打内部细节：异常文本可能带路径、SQL、上游 URL
            yield _sse_frame(
                {
                    "type": "error",
                    "message": "回答过程中出现异常，请重试；若反复出现请联系管理员。",
                    "code": type(exc).__name__,
                }
            )
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers=_SSE_HEADERS)
