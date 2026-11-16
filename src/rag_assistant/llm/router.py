"""LLM 路由：云端优先、本地兜底（由**服务端**决定，前端不参与）。

为什么需要它：
- 云端（DeepSeek）最准，但没 Key / 欠费 / 网络抖动时会失败；
- 本地（LM Studio）不花钱、离线可用，但慢且格式服从性差；
- 让服务端按"主 + 备"自动切换，比在前端放一个"选模型"下拉框更合理：
  终端用户不关心模型，运维也不希望有人绕过成本控制。

切换语义（只在 `LLMError` 时切换，不在"回答不好"时切换）：
- 主成功 → 用主；
- 主抛 LLMError（超时/鉴权/额度/连接被拒）→ 记 WARNING，改用备；
- 备也失败 → 抛原异常（带上两边信息）。
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

from ..exceptions import LLMError
from .base import LLMClient, LLMMessage, LLMResponse

logger = logging.getLogger("rag_assistant.llm.router")


class LLMRouter(LLMClient):
    """主备路由。:attr:`active` 记录最近一次真正被用到的后端，供 /health 暴露。"""

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self._primary = primary
        self._fallback = fallback
        self.active = getattr(primary, "model", "primary")
        self.fallback_used = 0

    @property
    def model(self) -> str:
        return f"{getattr(self._primary, 'model', '?')}→{getattr(self._fallback, 'model', '?')}"

    @property
    def is_mock(self) -> bool:
        return False

    def describe(self) -> dict[str, object]:
        return {
            "primary": getattr(self._primary, "model", "?"),
            "fallback": getattr(self._fallback, "model", "?"),
            "active": self.active,
            "fallback_used": self.fallback_used,
        }

    # ---- 调用 ----
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            response = self._primary.complete(messages, temperature=temperature, max_tokens=max_tokens)
            self.active = getattr(self._primary, "model", "primary")
            return response
        except LLMError as primary_error:
            logger.warning("主模型失败，切换备用模型：%s", primary_error)
            try:
                response = self._fallback.complete(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
            except LLMError as fallback_error:
                self.active = "failed"
                raise LLMError(
                    f"主模型与备用模型都失败。主：{primary_error}；备：{fallback_error}"
                ) from fallback_error
            self.active = getattr(self._fallback, "model", "fallback")
            self.fallback_used += 1
            return response

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        try:
            chunks = self._primary.stream(messages, temperature=temperature, max_tokens=max_tokens)
            first = next(chunks, None)
        except LLMError as primary_error:
            logger.warning("主模型流式失败，切换备用模型：%s", primary_error)
            yield from self._fallback.stream(messages, temperature=temperature, max_tokens=max_tokens)
            self.active = getattr(self._fallback, "model", "fallback")
            self.fallback_used += 1
            return

        self.active = getattr(self._primary, "model", "primary")
        if first:
            yield first
        yield from chunks
