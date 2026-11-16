"""DeepSeek / 本地 OpenAI 兼容服务（LM Studio、Ollama、vLLM）的统一客户端。

同一份实现同时服务两种目标：
- 云端 DeepSeek：``base_url=https://api.deepseek.com``，需要 ``DEEPSEEK_API_KEY``；
- 本地推理服务：``base_url=http://127.0.0.1:1234/v1``（LM Studio 默认），**不需要 Key**
  （OpenAI SDK 要求非空，这里塞占位串 ``not-needed``）。

实现要点：
- 用 ``langchain_openai.ChatOpenAI``，这是选 LangChain 的实际用途；
- **延迟构造**：只有真正调用时才建客户端，因此 mock 模式、无 Key 环境、
  单元测试都不会尝试联网；
- 用量口径：优先读 ``usage_metadata``（langchain-core 统一字段），
  退化读 ``response_metadata.token_usage``（OpenAI 原始字段）；
- 所有异常统一包成 :class:`~rag_assistant.exceptions.LLMError`，上层只处理一种错误。

需核实（已在本机 langchain-openai 1.7.3 上核对过构造参数）：
``model`` / ``base_url`` / ``api_key`` 是 pydantic 层的别名，映射到
``model_name`` / ``openai_api_base`` / ``openai_api_key``；``timeout`` 对应
``request_timeout``。升级 langchain 版本时请重跑 ``tests/test_llm.py``。
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Sequence

from ..config import LLMConfig
from ..exceptions import LLMError
from .base import LLMClient, LLMMessage, LLMResponse, LLMUsage

logger = logging.getLogger("rag_assistant.llm.deepseek")

_ROLE_MAP = {"system": "system", "user": "human", "assistant": "ai"}


class DeepSeekLLM(LLMClient):
    """DeepSeek 官方 API 客户端。"""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self.model = config.model
        self._client: Any | None = None

    # ---- 客户端 ----
    @property
    def configured(self) -> bool:
        """是否具备可用的凭据（本地服务不需要 Key，视为已配置）。"""
        return bool(self._config.api_key) or self._config.is_local

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.configured:
            raise LLMError(
                f"未配置 {self._config.api_key_env}，无法调用 {self._config.model}。"
                "请在 .env 中填写，或设置 RAG_FORCE_MOCK=1 使用 mock 模式。"
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("调用 OpenAI 兼容接口需要 langchain-openai：pip install langchain-openai") from exc

        logger.info(
            "初始化 LLM：model=%s base_url=%s（本地=%s）",
            self._config.model,
            self._config.base_url,
            self._config.is_local,
        )
        self._client = ChatOpenAI(
            model=self._config.model,
            base_url=self._config.base_url,
            # 本地服务通常不校验 Key，但 OpenAI SDK 要求非空
            api_key=self._config.api_key or ("not-needed" if self._config.is_local else ""),
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout_s,
            max_retries=self._config.max_retries,
            **({"model_kwargs": dict(self._config.model_kwargs)} if self._config.model_kwargs else {}),
        )
        return self._client

    # ---- 转换 ----
    @staticmethod
    def _to_langchain(messages: Sequence[LLMMessage]) -> list[tuple[str, str]]:
        """转成 langchain 的 (role, content) 元组序列。"""
        return [(_ROLE_MAP[message.role], message.content) for message in messages]

    @staticmethod
    def _usage_of(result: Any) -> LLMUsage:
        """提取 token 用量，兼容两种字段来源。"""
        usage_meta = getattr(result, "usage_metadata", None)
        if isinstance(usage_meta, dict) and usage_meta:
            return LLMUsage(
                prompt_tokens=int(usage_meta.get("input_tokens", 0) or 0),
                completion_tokens=int(usage_meta.get("output_tokens", 0) or 0),
            )
        raw = (getattr(result, "response_metadata", None) or {}).get("token_usage") or {}
        return LLMUsage(
            prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw.get("completion_tokens", 0) or 0),
        )

    @staticmethod
    def _reasoning_of(result: Any) -> str:
        """取出思考型模型的 reasoning 内容（只用于诊断，绝不当作答案）。"""
        for container in (
            getattr(result, "additional_kwargs", None),
            getattr(result, "response_metadata", None),
        ):
            if isinstance(container, dict):
                for key in ("reasoning_content", "reasoning"):
                    value = container.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        return ""

    # ---- 调用 ----
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        payload = self._to_langchain(messages)
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        runnable = client.bind(**overrides) if overrides else client
        try:
            result = runnable.invoke(payload)
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权/限流统一收口
            raise LLMError(f"LLM 调用失败（{self._config.model}）：{exc}") from exc

        content = getattr(result, "content", "")
        if isinstance(content, list):  # 某些版本会返回 content blocks
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        metadata = getattr(result, "response_metadata", None) or {}
        return LLMResponse(
            text=str(content),
            usage=self._usage_of(result),
            model=str(metadata.get("model_name", self._config.model)),
            finish_reason=str(metadata.get("finish_reason", "")),
            reasoning=self._reasoning_of(result),
        )

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        client = self._ensure_client()
        payload = self._to_langchain(messages)
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        runnable = client.bind(**overrides) if overrides else client
        try:
            for chunk in runnable.stream(payload):
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
                    )
                if content:
                    yield str(content)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM 流式调用失败（{self._config.model}）：{exc}") from exc

#: 别名：本地 OpenAI 兼容服务（LM Studio / Ollama / vLLM）用同一个实现
OpenAICompatLLM = DeepSeekLLM

