"""问题改写测试：JSON 解析、降级、变体去重与数量上限。"""

from __future__ import annotations

from typing import Sequence

import pytest

from rag_assistant.config import Settings
from rag_assistant.generation.rewriter import QueryRewriter, RewriteResult, _extract_json
from rag_assistant.llm.base import LLMClient, LLMMessage, LLMResponse, LLMUsage


class StubLLM(LLMClient):
    """返回固定文本的 LLM 桩。"""

    model = "stub"

    def __init__(self, text: str = "", *, fail: bool = False) -> None:
        self._text = text
        self._fail = fail
        self.calls = 0

    def complete(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None) -> LLMResponse:
        self.calls += 1
        if self._fail:
            raise RuntimeError("boom")
        return LLMResponse(text=self._text, usage=LLMUsage(prompt_tokens=10, completion_tokens=5), model=self.model)

    def stream(self, messages: Sequence[LLMMessage], *, temperature=None, max_tokens=None):
        yield self._text


@pytest.fixture()
def rewriter_settings(mock_settings: Settings) -> Settings:
    return mock_settings


# --------------------------------------------------------------------------
# JSON 抽取
# --------------------------------------------------------------------------
def test_extract_json_plain() -> None:
    assert _extract_json('{"standalone": "a", "variants": []}') == {"standalone": "a", "variants": []}


def test_extract_json_with_chatter_and_fence() -> None:
    text = '好的，结果如下：\n```json\n{"standalone": "休学期限", "variants": ["休学多长时间"]}\n```\n以上。'
    parsed = _extract_json(text)
    assert parsed is not None
    assert parsed["standalone"] == "休学期限"


def test_extract_json_invalid() -> None:
    assert _extract_json("抱歉，我无法回答") is None
    assert _extract_json("{broken") is None


# --------------------------------------------------------------------------
# 改写主流程
# --------------------------------------------------------------------------
def test_skip_when_no_history_and_multi_query_disabled(mock_settings: Settings) -> None:
    """首轮 + 关闭 multi-query：省一次调用。"""
    from dataclasses import replace

    disabled = replace(mock_settings, retrieval=replace(mock_settings.retrieval, enable_multi_query=False))
    llm = StubLLM('{"standalone": "不应被调用"}')
    result = QueryRewriter(disabled, llm).rewrite("休学多久", history_text="")
    assert llm.calls == 0
    assert result.skipped is True
    assert result.standalone == "休学多久"


def test_calls_llm_when_history_exists(mock_settings: Settings) -> None:
    llm = StubLLM('{"standalone": "休学期限", "variants": ["休学多长时间"]}')
    result = QueryRewriter(mock_settings, llm).rewrite("那期限呢", history_text="用户：休学多久\n助手：一年。")
    assert llm.calls == 1
    assert result.standalone == "休学期限"
    assert result.variants == ["休学多长时间"]
    assert result.degraded is False
    assert result.usage.total_tokens == 15


def test_variants_dedup_and_limit(mock_settings: Settings) -> None:
    """变体去重、剔除与 standalone 相同的项、按配置截断。"""
    limit = mock_settings.retrieval.multi_query
    variants = ["休学期限多久", "休学期限多久", "休学期限", "休学最长多久", "休学期限规定", "休学期限是几年"]
    payload = '{"standalone": "休学期限", "variants": ' + str(variants).replace("'", '"') + "}"
    result = QueryRewriter(mock_settings, StubLLM(payload)).rewrite("休学多久", history_text="用户：你好")
    assert result.variants == ["休学期限多久", "休学最长多久", "休学期限规定"][:limit]
    assert "休学期限" not in result.variants


def test_drifting_variants_dropped(mock_settings: Settings) -> None:
    """跑题的等价查询必须丢掉：RRF 里每路权重相同，跑偏会稀释召回。"""
    payload = '{"standalone": "休学期限", "variants": ["食堂几点开门", "休学最长多久"]}'
    result = QueryRewriter(mock_settings, StubLLM(payload)).rewrite("休学多久", history_text="用户：你好")
    assert result.variants == ["休学最长多久"]


def test_degraded_on_garbage_output(mock_settings: Settings) -> None:
    """模型回了一段人话而不是 JSON：要能降级，不能抛异常。"""
    llm = StubLLM("这个问题应该问教务处。")
    result = QueryRewriter(mock_settings, llm).rewrite("休学多久", history_text="用户：你好")
    assert result.degraded is True
    assert result.standalone == "休学多久"  # 回落到原问题


def test_partial_json_recovery(mock_settings: Settings) -> None:
    """坏 JSON 但能正则抽出 standalone 时，尽量用上它。"""
    llm = StubLLM('{"standalone": "复学所需材料", "variants": [oops}')
    result = QueryRewriter(mock_settings, llm).rewrite("那材料呢", history_text="用户：复学怎么办")
    assert result.degraded is True
    assert result.standalone == "复学所需材料"


def test_degraded_on_llm_error(mock_settings: Settings) -> None:
    llm = StubLLM(fail=True)
    result = QueryRewriter(mock_settings, llm).rewrite("休学多久", history_text="用户：你好")
    assert result.degraded is True
    assert result.standalone == "休学多久"


def test_empty_question(mock_settings: Settings) -> None:
    result = QueryRewriter(mock_settings, StubLLM("{}")).rewrite("   ", history_text="")
    assert result.standalone == ""
    assert result.skipped is True


def test_queries_helper() -> None:
    result = RewriteResult(standalone="A", variants=["B", "C"])
    assert result.queries == ["A", "B", "C"]


def test_real_mock_llm_end_to_end(mock_settings: Settings) -> None:
    """用真实 MockLLM 走一遍，确认 prompt 与解析器口径一致。"""
    from rag_assistant.llm import MockLLM

    rewriter = QueryRewriter(mock_settings, MockLLM())
    result = rewriter.rewrite("休学最长可以多久", history_text="")
    assert result.standalone
    assert result.degraded is False
