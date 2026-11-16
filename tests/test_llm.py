"""LLM 层测试：mock 任务分派、DeepSeek 客户端构造（不联网）。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rag_assistant.config import build_settings
from rag_assistant.exceptions import LLMError
from rag_assistant.llm import DeepSeekLLM, MockLLM, get_llm
from rag_assistant.llm.base import LLMMessage, LLMUsage
from rag_assistant.llm.prompts import answer_prompt, rewrite_prompt, summary_prompt, task_of

REFUSE = "当前知识库中未找到相关依据，建议咨询对应部门或人工客服。"

CONTEXT_BLOCK = "[1] 文档：学生学籍管理规定.md ｜ 章节：第三章 休学、复学与退学 > 第七条 休学条件 ｜ 页码：无"


# --------------------------------------------------------------------------
# MockLLM
# --------------------------------------------------------------------------
def test_task_marker_parsing() -> None:
    messages = rewrite_prompt("休学多久？", "", 2)
    assert task_of(messages) == "rewrite"
    assert messages[0].content.startswith("[[TASK:rewrite]]")


def test_mock_rewrite_returns_json() -> None:
    llm = MockLLM()
    result = llm.complete(rewrite_prompt("在读证明怎么办", "", 2))
    payload = json.loads(result.text)
    assert payload["standalone"] == "在读证明怎么办"
    assert isinstance(payload["variants"], list)
    assert llm.calls == ["rewrite"]


def test_mock_rewrite_uses_history_for_pronoun() -> None:
    """mock 的"指代消解"= 把上一问拼进来（规则实现，不代表语义能力）。"""
    llm = MockLLM()
    history = "用户：休学最长可以多久\n助手：休学期一般以一年为期。"
    result = llm.complete(rewrite_prompt("那它的期限呢？", history, 2))
    payload = json.loads(result.text)
    assert "休学最长可以多久" in payload["standalone"]


def test_mock_answer_cites_context() -> None:
    """回答里必须出现真实资料编号，否则闸门 2 会把 mock 自己也判死。"""
    llm = MockLLM(refuse_text=REFUSE)
    messages = answer_prompt(
        "休学最长多久",
        [f"{CONTEXT_BLOCK}\n正文：学生因病需停课治疗的应当办理休学。休学期一般以一年为期，累计不得超过两年。"],
        refuse_text=REFUSE,
    )
    text = llm.complete(messages).text
    assert "[1]" in text
    assert "应当办理休学" in text  # mock 抽取资料首句
    assert "学生学籍管理规定.md" in text


def test_mock_answer_without_context_refuses() -> None:
    llm = MockLLM(refuse_text=REFUSE)
    text = llm.complete(answer_prompt("食堂几点开门", [], refuse_text=REFUSE)).text
    assert text == REFUSE


def test_mock_answer_table_snippet_includes_data_row() -> None:
    """表格资料不能只引用表头——那样等于没回答。"""
    table = (
        "[1] 文档：差旅与会议费标准表.xlsx ｜ 章节：工作表：差旅住宿费标准 ｜ 页码：无\n"
        "正文：| 地区类别 | 住宿费限额 |\n"
        "|---|---|\n"
        "| 省内 | 500 |"
    )
    llm = MockLLM(refuse_text=REFUSE)
    text = llm.complete(answer_prompt("差旅住宿费标准", [table], refuse_text=REFUSE)).text
    assert "500" in text
    assert "[1]" in text


def test_mock_summary_truncates() -> None:
    llm = MockLLM()
    long_turns = "用户：问题一\n助手：回答一\n" * 40
    text = llm.complete(summary_prompt("", long_turns)).text
    assert 0 < len(text) <= 150


def test_mock_stream_concatenates() -> None:
    llm = MockLLM(refuse_text=REFUSE)
    messages = answer_prompt(
        "休学",
        [f"{CONTEXT_BLOCK}\n正文：休学期一般以一年为期。"],
        refuse_text=REFUSE,
    )
    streamed = "".join(llm.stream(messages))
    assert streamed == llm.complete(messages).text
    assert llm.is_mock is True


def test_mock_usage_reported() -> None:
    llm = MockLLM()
    response = llm.complete(rewrite_prompt("在读证明怎么办", "", 1))
    assert isinstance(response.usage, LLMUsage)
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.model == "mock-llm"


def test_mock_unknown_task_returns_empty() -> None:
    llm = MockLLM()
    assert llm.complete([LLMMessage(role="system", content="普通系统提示"), LLMMessage(role="user", content="你好")]).text == ""


# --------------------------------------------------------------------------
# 工厂
# --------------------------------------------------------------------------
def test_factory_mock(mock_settings: Any) -> None:
    llm = get_llm(mock_settings)
    assert isinstance(llm, MockLLM)
    assert llm.is_mock is True


def test_factory_without_key_falls_back_to_mock(base_config: dict[str, Any], project: Any) -> None:
    """没有 Key 时不应抛错，而是回落 mock（需求：无 Key 可跑通）。"""
    settings = build_settings(base_config, root=project, env={})
    assert settings.llm.is_mock is True
    assert isinstance(get_llm(settings), MockLLM)


def test_factory_with_key_is_deepseek(base_config: dict[str, Any], project: Any) -> None:
    settings = build_settings(base_config, root=project, env={"DEEPSEEK_API_KEY": "sk-test"})
    llm = get_llm(settings)
    assert isinstance(llm, DeepSeekLLM)
    assert llm.is_mock is False
    assert llm._client is None  # 延迟构造，构造期不联网


# --------------------------------------------------------------------------
# DeepSeek
# --------------------------------------------------------------------------
def test_deepseek_without_key_raises(base_config: dict[str, Any], project: Any) -> None:
    settings = build_settings(base_config, root=project, env={})
    llm = DeepSeekLLM(settings.llm)
    assert llm.configured is False
    with pytest.raises(LLMError, match="未配置"):
        llm.complete([LLMMessage(role="user", content="hi")])


def test_deepseek_role_mapping() -> None:
    messages = [
        LLMMessage(role="system", content="s"),
        LLMMessage(role="user", content="u"),
        LLMMessage(role="assistant", content="a"),
    ]
    assert DeepSeekLLM._to_langchain(messages) == [("system", "s"), ("human", "u"), ("ai", "a")]


def test_deepseek_client_construction(base_config: dict[str, Any], project: Any) -> None:
    """用真实 langchain-openai 构造客户端（不发请求），核对参数映射是否仍然有效。"""
    pytest.importorskip("langchain_openai")
    settings = build_settings(base_config, root=project, env={"DEEPSEEK_API_KEY": "sk-test"})
    llm = DeepSeekLLM(settings.llm)
    client = llm._ensure_client()
    assert client.model_name == settings.llm.model
    assert str(client.openai_api_base) == settings.llm.base_url
    assert llm._ensure_client() is client  # 只构造一次


def test_deepseek_usage_extraction() -> None:
    class FakeResult:
        usage_metadata = {"input_tokens": 11, "output_tokens": 7}
        response_metadata = {"model_name": "deepseek-chat", "finish_reason": "stop"}

    usage = DeepSeekLLM._usage_of(FakeResult())
    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 18

    class LegacyResult:
        usage_metadata = None
        response_metadata = {"token_usage": {"prompt_tokens": 3, "completion_tokens": 4}}

    legacy = DeepSeekLLM._usage_of(LegacyResult())
    assert legacy.total_tokens == 7


# --------------------------------------------------------------------------
# 本地 OpenAI 兼容服务（LM Studio / Ollama）：不需要 Key
# --------------------------------------------------------------------------
def test_local_provider_does_not_require_key(base_config: dict[str, Any], project: Any) -> None:
    settings = build_settings(
        {
            **base_config,
            "llm": {
                **base_config["llm"],
                "provider": "lmstudio",
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "local-model",
            },
        },
        root=project,
        env={},
    )
    assert settings.llm.is_local is True
    assert settings.llm.requires_key is False
    # 没有 Key 也不算 mock —— 这是与云端 provider 的关键区别
    assert settings.llm.is_mock is False
    llm = get_llm(settings)
    assert isinstance(llm, DeepSeekLLM)
    assert llm.is_mock is False


def test_local_provider_passes_placeholder_key(base_config: dict[str, Any], project: Any) -> None:
    pytest.importorskip("langchain_openai")
    settings = build_settings(
        {
            **base_config,
            "llm": {
                **base_config["llm"],
                "provider": "openai_compatible",
                "base_url": "http://127.0.0.1:1234/v1",
                "model": "local-model",
            },
        },
        root=project,
        env={},
    )
    llm = DeepSeekLLM(settings.llm)
    client = llm._ensure_client()
    assert client.model_name == "local-model"
    assert str(client.openai_api_base) == "http://127.0.0.1:1234/v1"


def test_model_kwargs_passthrough(base_config: dict[str, Any], project: Any) -> None:
    pytest.importorskip("langchain_openai")
    settings = build_settings(
        {
            **base_config,
            "llm": {
                **base_config["llm"],
                "provider": "lmstudio",
                "model_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        },
        root=project,
        env={},
    )
    client = DeepSeekLLM(settings.llm)._ensure_client()
    assert client.model_kwargs.get("chat_template_kwargs") == {"enable_thinking": False}


def test_reasoning_extraction() -> None:
    class WithReasoning:
        additional_kwargs = {"reasoning_content": "先分析问题，再作答…"}
        response_metadata: dict[str, Any] = {}

    class LegacyReasoning:
        additional_kwargs: dict[str, Any] = {}
        response_metadata = {"reasoning": "旧字段"}

    class Plain:
        additional_kwargs: dict[str, Any] = {}
        response_metadata: dict[str, Any] = {}

    assert DeepSeekLLM._reasoning_of(WithReasoning()).startswith("先分析")
    assert DeepSeekLLM._reasoning_of(LegacyReasoning()) == "旧字段"
    assert DeepSeekLLM._reasoning_of(Plain()) == ""
