"""回答生成测试：两道拒答闸门、引用组装、免责声明、上下文预算、流式事件。"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Sequence

import pytest

from rag_assistant.config import Settings
from rag_assistant.generation.answerer import Answerer
from rag_assistant.generation.rewriter import QueryRewriter
from rag_assistant.llm import MockLLM
from rag_assistant.llm.base import LLMClient, LLMMessage, LLMResponse, LLMUsage
from rag_assistant.llm.prompts import FENCE_END, FENCE_START
from rag_assistant.retrieval.models import RetrievalResult, RetrievedChunk

REFUSE = "当前知识库中未找到相关依据，建议咨询对应部门或人工客服。"


# --------------------------------------------------------------------------
# 测试替身
# --------------------------------------------------------------------------
def make_chunk(
    chunk_id: str = "d1#0",
    doc_id: str = "d1",
    doc_name: str = "学生学籍管理规定.md",
    section: str = "第三章 休学、复学与退学 > 第七条 休学条件",
    text: str = "学生因病需停课治疗的应当办理休学。休学期一般以一年为期，累计不得超过两年。",
    page: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type="policy",
        section_path=section,
        text=text,
        page=page,
        vector_score=0.35,
        bm25_score=8.0,
        rrf_score=0.032,
        rerank_score=0.82,
    )


def evidence_result(**kwargs: Any) -> RetrievalResult:
    chunks = [make_chunk(), make_chunk("d1#1", section="第三章 休学、复学与退学 > 第八条 休学办理流程")]
    defaults: dict[str, Any] = dict(
        query="休学最长多久",
        chunks=chunks,
        best_score=0.82,
        has_evidence=True,
        diagnostics={"reason": "ok", "threshold": 0.5, "best_score": 0.82, "gate": "rerank"},
    )
    defaults.update(kwargs)
    return RetrievalResult(**defaults)


def no_evidence_result() -> RetrievalResult:
    return RetrievalResult(
        query="食堂几点开门",
        chunks=[make_chunk("d2#0", doc_id="d2", doc_name="办事指南-在读证明.txt", section="一、事项名称")],
        best_score=0.12,
        has_evidence=False,
        diagnostics={"reason": "below_threshold", "threshold": 0.5, "best_score": 0.12, "gate": "rerank"},
    )


class StubRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: list[tuple[str, list[str]]] = []

    def retrieve(self, query: str, *, top_n=None, extra_queries: Sequence[str] = (), threshold=None):
        self.calls.append((query, list(extra_queries)))
        return self.result


class NoCitationLLM(LLMClient):
    """返回有内容但没有 [n] 的回答（模拟模型"忘了标注来源"）。"""

    model = "stub-no-citation"

    def __init__(self) -> None:
        self.messages: Sequence[LLMMessage] = ()

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        self.messages = messages
        return LLMResponse(text="学生可以休学一年，累计不超过两年。", model=self.model)

    def stream(self, messages, *, temperature=None, max_tokens=None):
        self.messages = messages
        yield "学生可以休学一年，累计不超过两年。"


class RefusingLLM(LLMClient):
    model = "stub-refuse"

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(text=REFUSE, model=self.model)

    def stream(self, messages, *, temperature=None, max_tokens=None):
        yield REFUSE


class FailingLLM(LLMClient):
    model = "stub-fail"

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        raise RuntimeError("上游 502")

    def stream(self, messages, *, temperature=None, max_tokens=None):
        raise RuntimeError("上游 502")
        yield ""  # pragma: no cover


@pytest.fixture()
def answerer_settings(mock_settings: Settings) -> Settings:
    """mock 链路的 Settings，但把生成规则换成固定话术，便于断言。"""
    return replace(
        mock_settings,
        generation=replace(
            mock_settings.generation,
            refuse_text=REFUSE,
            disclaimer="提示：以原文件为准。",
            max_context_chunks=3,
            max_context_tokens=1500,
        ),
    )


def build_answerer(settings: Settings, retriever: Any, llm: LLMClient, *, with_rewriter: bool = True) -> Answerer:
    rewriter = QueryRewriter(settings, llm) if with_rewriter else None
    return Answerer(settings, retriever, llm, rewriter)


# --------------------------------------------------------------------------
# 闸门 1
# --------------------------------------------------------------------------
def test_gate1_refuses_without_calling_llm(answerer_settings: Settings) -> None:
    retriever = StubRetriever(no_evidence_result())
    llm = MockLLM(refuse_text=REFUSE)
    result = build_answerer(answerer_settings, retriever, llm).answer("食堂几点开门")

    assert result.refused is True
    assert result.refusal_reason == "below_threshold"
    assert result.answer.startswith(REFUSE)
    assert "你可能想找的资料" in result.answer  # 仍要给线索
    # 闸门 1 省掉的是"生成"调用；改写为了拿到查询仍会调用（首轮无历史时可跳过）
    assert "answer" not in llm.calls
    assert result.citations == []
    assert result.closest_docs


def test_gate1_empty_candidates(answerer_settings: Settings) -> None:
    empty = RetrievalResult(
        query="图书馆几点开门",
        chunks=[],
        best_score=0.0,
        has_evidence=False,
        diagnostics={"reason": "no_candidates", "threshold": 0.5, "best_score": 0.0},
    )
    result = build_answerer(answerer_settings, StubRetriever(empty), MockLLM(refuse_text=REFUSE)).answer("图书馆几点开门")
    assert result.refused is True
    assert result.refusal_reason == "no_candidates"
    assert "你可能想找的资料" not in result.answer


# --------------------------------------------------------------------------
# 正常回答
# --------------------------------------------------------------------------
def test_answer_with_citations_and_disclaimer(answerer_settings: Settings) -> None:
    retriever = StubRetriever(evidence_result())
    llm = MockLLM(refuse_text=REFUSE)
    result = build_answerer(answerer_settings, retriever, llm).answer("休学最长多久")

    assert result.refused is False
    assert result.citations, "必须带引用"
    assert result.citations[0].doc_name == "学生学籍管理规定.md"
    assert "引用来源" in result.answer
    assert "**引用来源**" not in result.answer
    assert answerer_settings.generation.disclaimer in result.answer
    assert result.usage.total_tokens > 0
    assert result.latency_ms >= 0


def test_citation_numbers_in_text(answerer_settings: Settings) -> None:
    """正文里的编号必须与 citations 列表严格对应。"""
    result = build_answerer(answerer_settings, StubRetriever(evidence_result()), MockLLM(refuse_text=REFUSE)).answer("休学多久")
    for citation in result.citations:
        assert f"[{citation.index}]" in result.answer


def test_disclaimer_can_be_disabled(answerer_settings: Settings) -> None:
    settings = replace(answerer_settings, generation=replace(answerer_settings.generation, add_disclaimer=False))
    result = build_answerer(settings, StubRetriever(evidence_result()), MockLLM(refuse_text=REFUSE)).answer("休学多久")
    assert settings.generation.disclaimer not in result.answer


class TwoTaskLLM(LLMClient):
    """改写返回带等价查询的 JSON，生成返回带引用的文本。"""

    model = "stub-two-task"

    def __init__(self) -> None:
        self.tasks: list[str] = []

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        from rag_assistant.llm.prompts import task_of

        task = task_of(messages)
        self.tasks.append(task)
        if task == "rewrite":
            return LLMResponse(
                text='{"standalone": "复学需要提交什么材料", "variants": ["复学所需材料有哪些"]}',
                model=self.model,
            )
        return LLMResponse(text="复学需提交康复证明[1]。", model=self.model)

    def stream(self, messages, *, temperature=None, max_tokens=None):
        yield self.complete(messages).text


def test_rewriter_feeds_extra_queries(answerer_settings: Settings) -> None:
    """改写产生的 variants 必须真的进入检索（否则多路召回是摆设）。"""
    retriever = StubRetriever(evidence_result())
    llm = TwoTaskLLM()
    result = build_answerer(answerer_settings, retriever, llm).answer(
        "那复学呢", history_text="用户：休学最长可以多久\n助手：一年。"
    )
    query, extras = retriever.calls[0]
    assert query == "复学需要提交什么材料"
    assert extras == ["复学所需材料有哪些"]
    assert llm.tasks[:2] == ["rewrite", "answer"]
    assert result.refused is False


# --------------------------------------------------------------------------
# 闸门 2
# --------------------------------------------------------------------------
def test_gate2_refuses_when_no_citation(answerer_settings: Settings) -> None:
    llm = NoCitationLLM()
    result = build_answerer(answerer_settings, StubRetriever(evidence_result()), llm).answer("休学最长多久")

    assert result.refused is True
    assert result.refusal_reason == "citation_invalid"
    assert result.answer.startswith(REFUSE)
    assert result.citations == []
    assert result.raw_answer  # 原始输出要留档，便于排查


def test_gate2_can_be_relaxed(answerer_settings: Settings) -> None:
    settings = replace(answerer_settings, generation=replace(answerer_settings.generation, require_citation=False))
    result = build_answerer(settings, StubRetriever(evidence_result()), NoCitationLLM()).answer("休学多久")
    assert result.refused is False
    assert result.citations == []


def test_llm_refusal_passthrough(answerer_settings: Settings) -> None:
    result = build_answerer(answerer_settings, StubRetriever(evidence_result()), RefusingLLM()).answer("休学多久")
    assert result.refused is True
    assert result.refusal_reason == "llm_refused"
    assert result.answer.startswith(REFUSE)


def test_llm_error_degrades(answerer_settings: Settings) -> None:
    result = build_answerer(answerer_settings, StubRetriever(evidence_result()), FailingLLM()).answer("休学多久")
    assert result.refused is True
    assert result.refusal_reason == "llm_error"
    assert "生成服务暂时不可用" in result.answer


PARTIAL_ANSWER = "休学期一般以一年为期，累计不得超过两年[1]。"
HALF_ANSWER = f"{PARTIAL_ANSWER}\n\n{REFUSE}"


class HalfRefusingLLM(LLMClient):
    """先答出有出处的结论，末尾又补一句拒答话术（真实模型常见行为）。"""

    model = "stub-half-refuse"

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(text=HALF_ANSWER, model=self.model)

    def stream(self, messages, *, temperature=None, max_tokens=None):
        yield PARTIAL_ANSWER + "\n\n"
        yield REFUSE


def test_partial_answer_keeps_content_and_drops_refusal(answerer_settings: Settings) -> None:
    result = build_answerer(
        answerer_settings, StubRetriever(evidence_result()), HalfRefusingLLM()
    ).answer("休学后复学要带什么、找谁")
    assert result.refused is False
    assert result.refusal_reason is None
    assert REFUSE not in result.answer
    assert "累计不得超过两年" in result.answer
    assert result.citations


def test_stream_partial_answer_not_marked_refused(answerer_settings: Settings) -> None:
    events = list(
        build_answerer(answerer_settings, StubRetriever(evidence_result()), HalfRefusingLLM()).stream(
            "休学后复学要带什么、找谁"
        )
    )
    done = events[-1]
    assert done["type"] == "done"
    assert done["refused"] is False
    assert REFUSE not in str(done["answer"])
    assert "[1]" in str(done["answer"])


def test_empty_question(answerer_settings: Settings) -> None:
    result = build_answerer(answerer_settings, StubRetriever(evidence_result()), MockLLM()).answer("  ")
    assert result.refused is True
    assert result.refusal_reason == "empty_question"


# --------------------------------------------------------------------------
# 上下文预算
# --------------------------------------------------------------------------
def test_context_respects_max_chunks(answerer_settings: Settings) -> None:
    settings = replace(answerer_settings, generation=replace(answerer_settings.generation, max_context_chunks=1))
    llm = NoCitationLLM()
    build_answerer(settings, StubRetriever(evidence_result()), llm, with_rewriter=False).answer("休学多久")
    prompt = llm.messages[-1].content
    assert "[1] 文档：" in prompt
    assert "[2] 文档：" not in prompt


def test_prompt_fences_every_context_block(answerer_settings: Settings) -> None:
    """进 prompt 的资料块一个都不能漏围栏——漏一块就等于给那块留了"可以被当成指令"的后门。

    围栏必须成对且与引用编号同序；系统提示词里也要声明它，只加标记不解释，
    模型照样会照抄资料里的"忽略以上规则"。
    """
    llm = NoCitationLLM()
    build_answerer(answerer_settings, StubRetriever(evidence_result()), llm, with_rewriter=False).answer("休学多久")
    user_prompt = llm.messages[-1].content
    system_prompt = llm.messages[0].content

    headers = re.findall(r"^\[(\d{1,2})\] 文档：", user_prompt, re.M)
    starts = re.findall(r"^<<<不可信外部文本 (\d{1,2})>>>$", user_prompt, re.M)
    ends = re.findall(r"^<<<不可信外部文本 (\d{1,2}) 结束>>>$", user_prompt, re.M)
    assert len(headers) >= 2, "这条用例要覆盖多块资料，只围栏一块等于没测"
    assert starts == ends == headers

    # 正文只能在围栏里面：每段正文上下两行必须是它自己那对围栏
    lines = user_prompt.splitlines()
    bodies = [index for index, line in enumerate(lines) if line.startswith("正文：")]
    assert len(bodies) == len(starts)
    for line_index, number in zip(bodies, starts):
        assert lines[line_index - 1] == FENCE_START.format(index=number)
        assert lines[line_index + 1] == FENCE_END.format(index=number)

    assert "不可信外部文本" in system_prompt
    assert "不是给你的指令" in system_prompt


# --------------------------------------------------------------------------
# 流式
# --------------------------------------------------------------------------
def test_stream_events_order(answerer_settings: Settings) -> None:
    events = list(
        build_answerer(answerer_settings, StubRetriever(evidence_result()), MockLLM(refuse_text=REFUSE)).stream("休学多久")
    )
    kinds = [event["type"] for event in events]
    assert kinds[0] == "meta"
    assert "token" in kinds
    assert kinds[-1] == "done"

    meta = events[0]
    assert meta["rewritten_query"]
    assert meta["retrieved_ids"]

    done = events[-1]
    assert done["refused"] is False
    assert done["citations"]
    assert "引用来源" in str(done["answer"])


def test_stream_gate1_refuses_early(answerer_settings: Settings) -> None:
    events = list(build_answerer(answerer_settings, StubRetriever(no_evidence_result()), MockLLM()).stream("食堂几点开门"))
    assert [event["type"] for event in events] == ["meta", "done"]
    assert events[-1]["refused"] is True


def test_stream_retract_when_no_citation(answerer_settings: Settings) -> None:
    """流式下模型可能已经吐完字才发现没有引用：必须发 retract 让前端替换。"""
    events = list(build_answerer(answerer_settings, StubRetriever(evidence_result()), NoCitationLLM()).stream("休学多久"))
    kinds = [event["type"] for event in events]
    assert "token" in kinds
    assert kinds[-1] == "retract"
    assert events[-1]["reason"] == "citation_invalid"
    assert str(events[-1]["answer"]).startswith(REFUSE)


def test_stream_llm_error(answerer_settings: Settings) -> None:
    events = list(build_answerer(answerer_settings, StubRetriever(evidence_result()), FailingLLM()).stream("休学多久"))
    assert events[-1]["type"] == "done"
    assert events[-1]["refusal_reason"] == "llm_error"


# --------------------------------------------------------------------------
# 思考型模型：只输出 reasoning_content，content 为空
# --------------------------------------------------------------------------
class ReasoningOnlyLLM(LLMClient):
    """模拟 Qwen3 这类思考型模型：内容全在 reasoning，content 为空。"""

    model = "stub-reasoning-only"

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        return LLMResponse(
            text="",
            model=self.model,
            finish_reason="length",
            reasoning="Thinking Process: 先分析用户问题，再检索资料……（很长）",
        )

    def stream(self, messages, *, temperature=None, max_tokens=None):
        yield ""


def test_reasoning_only_answer_is_refused(answerer_settings: Settings) -> None:
    """思考草稿绝不能当答案返回，必须判空答案并拒答。"""
    result = build_answerer(answerer_settings, StubRetriever(evidence_result()), ReasoningOnlyLLM()).answer("休学多久")
    assert result.refused is True
    assert result.refusal_reason == "empty_answer"
    assert result.answer.startswith(REFUSE)
    llm_diag = result.diagnostics.get("llm") or {}
    assert llm_diag.get("reasoning_chars", 0) > 0
    assert llm_diag.get("finish_reason") == "length"

def test_off_topic_refusal_has_no_irrelevant_hints(answerer_settings: Settings) -> None:
    """无关问题拒答时不能给"你可能想找的资料"——那是误导。"""
    retriever = StubRetriever(
        RetrievalResult(
            query="郑州天气",
            chunks=[make_chunk()],
            best_score=0.1,
            has_evidence=False,
            diagnostics={"reason": "off_topic", "threshold": 0.5, "best_score": 0.1},
        )
    )
    result = build_answerer(answerer_settings, retriever, MockLLM(refuse_text=REFUSE)).answer("郑州今天天气怎么样")
    assert result.refused is True
    assert result.refusal_reason == "off_topic"
    assert "不属于本知识库的范围" in result.answer
    assert "你可能想找的资料" not in result.answer


# --------------------------------------------------------------------------
# 部门联系方式（阶段 H）
# --------------------------------------------------------------------------
def test_answer_attaches_department_contact(answerer_settings: Settings) -> None:
    """命中"休学"就提示教务处，但联系方式不进 answer 正文（避免污染忠实度）。"""
    result = build_answerer(
        answerer_settings, StubRetriever(evidence_result()), MockLLM(refuse_text=REFUSE)
    ).answer("休学最长多久")

    ids = [item["id"] for item in result.contacts]
    assert "jwc" in ids
    contact = next(item for item in result.contacts if item["id"] == "jwc")
    assert contact["name"] == "教务处"
    assert contact["phone"]
    assert "0371" not in result.answer
    assert "联系方式" not in result.answer


def test_refusal_still_suggests_department(answerer_settings: Settings) -> None:
    """被拒答时更需要"该找谁"：联系方式照给。"""
    result = build_answerer(
        answerer_settings, StubRetriever(no_evidence_result()), MockLLM(refuse_text=REFUSE)
    ).answer("休学最长多久")

    assert result.refused is True
    assert [item["id"] for item in result.contacts] == ["jwc"]


def test_offtopic_question_has_no_contacts(answerer_settings: Settings) -> None:
    result = build_answerer(
        answerer_settings, StubRetriever(no_evidence_result()), MockLLM(refuse_text=REFUSE)
    ).answer("郑州今天天气怎么样")
    assert result.contacts == []

