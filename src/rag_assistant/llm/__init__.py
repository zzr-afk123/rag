"""LLM 工厂：云端（DeepSeek）/ 本地（LM Studio、Ollama）/ 主备路由 / Mock。"""

from __future__ import annotations

from dataclasses import replace

from ..config import Settings
from .base import LLMClient, LLMMessage, LLMResponse, LLMUsage, assistant_message, system_message, user_message
from .deepseek import DeepSeekLLM, OpenAICompatLLM
from .mock import MockLLM
from .router import LLMRouter

__all__ = [
    "DeepSeekLLM",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMRouter",
    "LLMUsage",
    "MockLLM",
    "OpenAICompatLLM",
    "assistant_message",
    "get_llm",
    "llm_note",
    "system_message",
    "user_message",
]


def _build_fallback(settings: Settings) -> OpenAICompatLLM | None:
    """按 ``llm.fallback_*`` 构造备用客户端（通常指向本地服务，无需 Key）。"""
    config = settings.llm
    if not config.has_fallback:
        return None
    fallback_config = replace(
        config,
        provider=config.fallback_provider,
        base_url=config.fallback_base_url or config.base_url,
        model=config.fallback_model,
        api_key="",
        fallback_provider="",
        fallback_base_url="",
        fallback_model="",
    )
    return OpenAICompatLLM(fallback_config)


def _tag(client: LLMClient, note: str) -> LLMClient:
    """给客户端挂一条"为什么是这个模型"的说明。

    运维台/健康检查里只显示 ``mock-llm`` 的时候，值班的人第一反应是"我是不是配错了"，
    却看不出错在哪。这条说明跟着客户端走，页面直接展示出来。
    """
    try:
        setattr(client, "note", note)
    except Exception:  # pragma: no cover - 只影响展示，不能影响启动
        pass
    return client


def llm_note(client: LLMClient) -> str:
    """取出 ``_tag`` 写的说明（路由器取主模型那条）。"""
    return str(getattr(client, "note", "") or "")


def get_llm(settings: Settings) -> LLMClient:
    """按配置返回 LLM（服务端决定用云端还是本地，前端不参与）。

    判定顺序：
    1. ``llm.provider == mock``（含 ``RAG_FORCE_MOCK=1``）→ :class:`MockLLM`
    2. 主模型可用 + 配了备用 → :class:`LLMRouter`（主失败自动切备）
    3. 主模型没有 Key 但配了备用（典型：无 DeepSeek Key，用本地）→ 直接用备用
    4. 只有主 → 主
    5. 都没有 → 记 WARNING 回落 :class:`MockLLM`（需求要求"无 Key 可跑通"）
    """
    config = settings.llm
    if config.provider.strip().lower() == "mock":
        return _tag(
            MockLLM(refuse_text=settings.generation.refuse_text),
            "配置为 mock（llm.provider=mock 或 RAG_FORCE_MOCK=1），回答由规则拼接，不调用模型",
        )

    fallback = _build_fallback(settings)
    primary: OpenAICompatLLM | None = None
    if config.requires_key and not config.api_key:
        import logging

        logging.getLogger("rag_assistant.llm").warning(
            "未检测到 %s，主模型不可用", config.api_key_env
        )
    else:
        primary = OpenAICompatLLM(config)

    if primary and fallback:
        return LLMRouter(primary, fallback)
    if fallback and not primary:
        import logging

        logging.getLogger("rag_assistant.llm").warning(
            "主模型无 Key，直接使用备用模型（本地）：%s", config.fallback_model
        )
        return _tag(
            fallback,
            f"未配置 {config.api_key_env}，按 llm.fallback_* 使用本地备用模型 {config.fallback_model}",
        )
    if primary:
        return primary

    import logging

    logging.getLogger("rag_assistant.llm").warning(
        "既没有可用的主模型 Key，也没有配置备用模型，使用 mock LLM（回答为规则生成，仅用于演示）"
    )
    return _tag(
        MockLLM(refuse_text=settings.generation.refuse_text),
        f"没有 {config.api_key_env}，也没有配置 llm.fallback_*（本地模型）——回落到 mock，"
        "回答是规则拼接的原文摘录；用「联网模型.bat」或「RAG 一键公网」启动即可接本地模型",
    )
