"""API 层：请求/响应模型（pydantic v2）。

只做"形状"约束与文档化，业务规则仍在 ``chat_service`` / ``answerer`` 里。
校验失败统一走 FastAPI 默认的 422，错误响应结构与 FastAPI 一致。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .. import __version__


class ChatRequest(BaseModel):
    """一轮提问。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"session_id": "demo-1", "question": "在读证明怎么开？", "threshold": 0.5}
        }
    )

    session_id: str = Field(..., min_length=1, max_length=64, description="会话 ID，相同 ID 共享上下文")
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="仅管理员可用的阈值覆盖；其他角色一律忽略，用 server 配置的 "
        "retrieval.score_threshold（防把拒答闸门关掉）",
    )


class CitationOut(BaseModel):
    """引用来源。"""

    index: int = Field(..., description="答案正文中的 [n] 编号")
    doc_id: str
    doc_name: str
    section_path: str = Field("", description="章节路径，如 第三章 > 第七条")
    page: int = Field(0, description="页码，0 表示无页码")
    chunk_id: str
    quote: str = Field("", description="原文摘录")
    file_version: str = ""
    doc_type: str = ""


class ContactOut(BaseModel):
    """命中部门（含联系方式）。

    与 ``citations`` 并列返回、**不写进 answer 正文**：联系方式属于"下一步该找谁"，
    不是回答内容，混进正文会污染忠实度评测，也容易被误读成制度条款。
    """

    id: str
    name: str
    full_name: str = ""
    phone: str = ""
    email: str = ""
    address: str = ""
    hours: str = ""
    note: str = ""


class UsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatResponse(BaseModel):
    """一轮问答结果。"""

    session_id: str
    answer: str
    citations: list[CitationOut] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    rewritten_query: str = ""
    extra_queries: list[str] = Field(default_factory=list)
    closest_docs: list[dict[str, Any]] = Field(default_factory=list)
    contacts: list[ContactOut] = Field(
        default_factory=list, description="问题涉及部门及其联系方式（命中关键词才有）"
    )
    latency_ms: float = 0.0
    usage: UsageOut = Field(default_factory=UsageOut)

    @classmethod
    def from_reply(cls, reply: Any) -> "ChatResponse":
        """从 ``ChatReply`` 转换（避免 API 层直接依赖 dataclass 字段顺序）。"""
        return cls(
            session_id=reply.session_id,
            answer=reply.answer,
            citations=[CitationOut(**citation.to_dict()) for citation in reply.citations],
            refused=reply.refused,
            refusal_reason=reply.refusal_reason,
            rewritten_query=reply.rewritten_query,
            extra_queries=list(reply.extra_queries),
            closest_docs=list(reply.closest_docs),
            contacts=[ContactOut(**dict(item)) for item in getattr(reply, "contacts", []) or []],
            latency_ms=round(reply.latency_ms, 1),
            usage=UsageOut(
                prompt_tokens=reply.usage_prompt,
                completion_tokens=reply.usage_completion,
            ),
        )


class SyncRequest(BaseModel):
    """知识库同步请求。"""

    rebuild: bool = Field(False, description="True 表示清空索引后全量重建")
    wait: bool = Field(False, description="True 表示同步执行并直接返回结果（调试/CI 用）")


class SyncAccepted(BaseModel):
    """已接受（202）。"""

    accepted: bool = True
    job_id: str
    status: str = "pending"
    message: str = ""


class SyncCompleted(BaseModel):
    """同步执行完成（200，仅 wait=true）。"""

    accepted: bool = True
    job_id: str
    status: str
    report: dict[str, Any] | None = None
    error: str | None = None


class KBStatus(BaseModel):
    """/kb/status 响应。"""

    documents: int = 0
    manifest_chunks: int = 0
    vector_chunks: int = 0
    bm25_chunks: int = 0
    last_sync: str | None = None
    collection: str = ""
    embedder: str = ""
    embedder_mock: bool = False
    reranker: str | None = None
    score_threshold: float = 0.0
    syncing: bool = False
    last_job_id: str | None = None
    last_error: str | None = None
    last_report: dict[str, Any] | None = None
    # ---- 自动化状态（阶段 E）----
    scheduler_enabled: bool = False
    next_sync_at: str | None = None
    watcher_active: bool = False
    lock: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """/health 响应。"""

    status: str = "ok"
    version: str = __version__
    llm: str = ""
    llm_mock: bool = False
    embedder: str = ""
    embedder_mock: bool = False
    chunks: int = 0
    score_threshold: float = 0.0
    #: 服务器模式（bind 0.0.0.0）下的局域网访问地址，供手机端直接打开
    lan_urls: list[str] = Field(default_factory=list)
    host: str = ""
    port: int = 0
    #: 主备路由信息：主模型 / 备用模型 / 最近一次实际使用的后端
    llm_fallback: str = ""
    llm_active: str = ""
    #: 为什么是当前这个模型（尤其是回落到 mock 的原因），给值班的人看
    llm_note: str = ""
    #: 配置里的对外域名（server.public_url），未配置时为空
    public_url: str = ""
