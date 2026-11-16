"""LLM 抽象层。

为什么要自己包一层而不是全用 LangChain 的 chain：
- 只有"发消息、拿文本、拿 token 用量、流式吐字"这四件事；
- mock 实现要能完全离线跑测试，不能依赖 LCEL 的运行时；
- 上层（改写、生成、摘要、评测）需要统一的 usage 统计口径。

DeepSeek 走 OpenAI 兼容协议，因此真实实现用 ``langchain_openai.ChatOpenAI``
（这也是选 LangChain 的实际用途之一）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    """一条对话消息。"""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMUsage:
    """token 用量（用于日志与成本核算）。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    """一次完整调用结果。

    ``reasoning`` 单独放：思考型模型（Qwen3 系、DeepSeek-R1 等）会把内容写进
    ``reasoning_content``，而 ``content`` 可能为空。思考内容**不能当答案用**：
    它没经过引用约束，直接展示等于把内部草稿交给用户。这里只用于诊断与告警。
    """

    text: str
    usage: LLMUsage = LLMUsage()
    model: str = ""
    finish_reason: str = ""
    reasoning: str = ""


class LLMClient(ABC):
    """LLM 接口。"""

    model: str = ""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """同步调用，返回完整文本。"""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式调用，逐段产出文本增量。"""

    @property
    def is_mock(self) -> bool:
        return False

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"<{type(self).__name__} model={self.model}>"


def system_message(content: str) -> LLMMessage:
    return LLMMessage(role="system", content=content)


def user_message(content: str) -> LLMMessage:
    return LLMMessage(role="user", content=content)


def assistant_message(content: str) -> LLMMessage:
    return LLMMessage(role="assistant", content=content)
