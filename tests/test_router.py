"""主备路由测试：云端失败自动切本地，且统计可观测。"""

from __future__ import annotations

from typing import Iterator, Sequence

import pytest

from rag_assistant.exceptions import LLMError
from rag_assistant.llm.base import LLMClient, LLMMessage, LLMResponse, LLMUsage
from rag_assistant.llm.router import LLMRouter


class OkLLM(LLMClient):
    def __init__(self, name: str, text: str = "ok") -> None:
        self.model = name
        self._text = text

    def complete(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(text=self._text, model=self.model, usage=LLMUsage(prompt_tokens=1, completion_tokens=1))

    def stream(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> Iterator[str]:
        yield self._text


class FailLLM(LLMClient):
    model = "failing"

    def complete(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> LLMResponse:
        raise LLMError("额度用尽")

    def stream(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> Iterator[str]:
        raise LLMError("连接被拒绝")
        yield ""  # pragma: no cover


class StreamFailAfterFirst(LLMClient):
    """首块正常、后续抛错：验证"流式无法中途重试"的真实行为。"""

    model = "half-broken"

    def complete(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(text="x")

    def stream(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> Iterator[str]:
        yield "第一段"
        raise LLMError("生成中途断开")


def test_primary_success() -> None:
    router = LLMRouter(OkLLM("cloud", "云端答案"), OkLLM("local", "本地答案"))
    result = router.complete([])
    assert result.text == "云端答案"
    assert router.active == "cloud"
    assert router.fallback_used == 0


def test_fallback_on_primary_error() -> None:
    router = LLMRouter(FailLLM(), OkLLM("local", "本地答案"))
    result = router.complete([])
    assert result.text == "本地答案"
    assert router.active == "local"
    assert router.fallback_used == 1


def test_both_failing_raises_with_both_messages() -> None:
    router = LLMRouter(FailLLM(), FailLLM())
    with pytest.raises(LLMError) as excinfo:
        router.complete([])
    assert "主模型与备用模型都失败" in str(excinfo.value)


def test_stream_falls_back_before_first_token() -> None:
    router = LLMRouter(FailLLM(), OkLLM("local", "本地流式"))
    assert "".join(router.stream([])) == "本地流式"
    assert router.fallback_used == 1


def test_stream_does_not_retry_after_first_token() -> None:
    """首块已发出后无法重试（否则前端会看到重复内容）——错误必须如实抛出。"""
    router = LLMRouter(StreamFailAfterFirst(), OkLLM("local", "不该出现"))
    chunks: list[str] = []
    with pytest.raises(LLMError):
        for chunk in router.stream([]):
            chunks.append(chunk)
    assert chunks == ["第一段"]


def test_describe_reports_both_sides() -> None:
    router = LLMRouter(OkLLM("cloud"), OkLLM("local"))
    info = router.describe()
    assert info["primary"] == "cloud"
    assert info["fallback"] == "local"
    assert info["fallback_used"] == 0
