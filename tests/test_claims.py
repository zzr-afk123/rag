"""claim 级引用校验测试：切句、覆盖判定、数字硬校验、改标、删句、降级。

这些用例直接复刻真实故障场景（第 9 题的"引用错标"、金额被改写），
所以断言里出现的中文都是制度语料里的原句。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import pytest

from rag_assistant.config import Settings
from rag_assistant.generation.answerer import Answerer
from rag_assistant.generation.claims import ClaimChecker, Lexicon, _numbers
from rag_assistant.generation.context import ContextBlock
from rag_assistant.retrieval.models import RetrievalResult, RetrievedChunk

REFUSE = "当前知识库中未找到相关依据，建议咨询对应部门或人工客服。"


def chunk(
    chunk_id: str = "d1#0",
    doc_id: str = "d1",
    doc_name: str = "办事指南-在读证明.txt",
    section: str = "二、办理流程",
    text: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type="guide",
        section_path=section,
        text=text,
        page=0,
        vector_score=0.3,
        bm25_score=1.0,
        rrf_score=0.02,
        rerank_score=0.8,
    )


def block(index: int, item: RetrievedChunk) -> ContextBlock:
    header = f"[{index}] 文档：{item.doc_name} ｜ 章节：{item.section_path} ｜ 页码：无页码"
    return ContextBlock(index=index, chunk=item, header=header, body=item.text)


GUIDE_TEXT = "在读证明由教务处负责办理，办理地点为学生事务大厅，办理时限为 3 个工作日。"
POLICY_TEXT = "学院教务处是学籍主管部门，负责学籍异动审核。"


@pytest.fixture()
def blocks() -> list[ContextBlock]:
    return [
        block(1, chunk(text=GUIDE_TEXT)),
        block(2, chunk("d2#0", "d2", "学生学籍管理规定.md", "第二章 学籍管理", POLICY_TEXT)),
    ]


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def test_numbers_normalized() -> None:
    assert _numbers("5000 元") == {"5000"}
    assert _numbers("5000.0 元") == {"5000"}
    assert _numbers("０．５") == {"0.5"}


def test_lexicon_cjk_words_kept(blocks: list[ContextBlock]) -> None:
    lexicon = Lexicon()
    tokens = lexicon.tokens("教务处是学籍主管部门")
    assert "教务处" in tokens
    assert "是" not in tokens  # 单字虚词被丢弃


# --------------------------------------------------------------------------
# 支持 / 不支撑
# --------------------------------------------------------------------------
def test_supported_claim_untouched(blocks: list[ContextBlock]) -> None:
    checker = ClaimChecker()
    text = "在读证明由教务处负责办理 [1]。"
    result = checker.verify(text, blocks)
    assert result.text == text
    assert result.checked == 1
    assert result.supported == 1
    assert result.corrections == []


def test_low_coverage_no_better_block_kept_in_fix_mode(blocks: list[ContextBlock]) -> None:
    """只有一条资料块、句子与它无关：fix 模式保留原样（不删句），只记诊断。"""
    checker = ClaimChecker()
    text = "学生可以申请国家奖学金 [1]。"
    result = checker.verify(text, blocks[:1])
    assert result.text == text
    assert result.dropped == []
    assert result.verdicts[0].supported is False
    assert result.to_dict()["unsupported"]


def test_number_guard_flags_rewritten_amount(blocks: list[ContextBlock]) -> None:
    """原文 3 个工作日被写成 5 个工作日 —— 覆盖率很高但数字对不上，必须判不支撑。"""
    checker = ClaimChecker(mode="strict")
    result = checker.verify("办理时限为 5 个工作日 [1]。", blocks)
    verdict = result.verdicts[0]
    assert verdict.supported is False
    assert verdict.reason == "missing_number"
    assert verdict.missing_numbers == ("5",)
    # 另一块也没有 5，改标救不回来 → strict 删句 → 正文为空
    assert result.dropped
    assert result.text == ""


def test_number_guard_accepts_same_number(blocks: list[ContextBlock]) -> None:
    checker = ClaimChecker()
    result = checker.verify("办理时限为 3 个工作日 [1]。", blocks)
    assert result.verdicts[0].supported is True


# --------------------------------------------------------------------------
# 改标（第 9 题的真实故障）
# --------------------------------------------------------------------------
def test_mislabeled_claim_reassigned(blocks: list[ContextBlock]) -> None:
    text = "在读证明由教务处负责办理 [1]。教务处是学籍主管部门 [1]。"
    checker = ClaimChecker(mode="fix")
    result = checker.verify(text, blocks)

    assert result.changed is True
    assert "教务处是学籍主管部门 [2]" in result.text
    assert len(result.corrections) == 1
    assert result.corrections[0].markers == (1,)
    assert result.corrections[0].corrected == (2,)
    # 原本正确的句子不能被改
    assert "在读证明由教务处负责办理 [1]" in result.text


def test_reassign_requires_clear_margin() -> None:
    """改标必须"明显更优"：覆盖差距达不到 margin 时不动。"""
    blocks = [
        block(1, chunk("d1#0", "d1", "学生学籍管理规定.md", "第七条 休学条件", "休学期一般以一年为期。")),
        block(2, chunk("d2#0", "d2", "学生学籍管理规定.md", "第八条 复学办理", "休学期一般以一年为期，累计不得超过两年。")),
    ]
    text = "休学期一般以一年为期，累计不得超过两年 [1]。"

    strict_margin = ClaimChecker(mode="fix", reassign_margin=0.6).verify(text, blocks)
    assert strict_margin.corrections == []
    assert strict_margin.text == text

    loose_margin = ClaimChecker(mode="fix", reassign_margin=0.15).verify(text, blocks)
    assert [item.corrected for item in loose_margin.corrections] == [(2,)]
    assert text.replace("[1]", "[2]") == loose_margin.text


def test_warn_mode_never_changes_text(blocks: list[ContextBlock]) -> None:
    text = "在读证明由教务处负责办理 [1]。教务处是学籍主管部门 [1]。"
    result = ClaimChecker(mode="warn").verify(text, blocks)
    assert result.text == text
    assert result.corrections  # 诊断里仍记录"本该改成 [2]"
    assert result.dropped == []


def test_off_mode_skips(blocks: list[ContextBlock]) -> None:
    result = ClaimChecker(mode="off").verify("随便 [1]。", blocks)
    assert result.enabled is False
    assert result.text == "随便 [1]。"


# --------------------------------------------------------------------------
# 切句 / 继承 / 结构保持
# --------------------------------------------------------------------------
def test_claim_without_marker_inherits_previous(blocks: list[ContextBlock]) -> None:
    text = "在读证明由教务处负责办理 [1]。办理地点为学生事务大厅。"
    result = ClaimChecker().verify(text, blocks)
    assert result.checked == 2
    assert result.verdicts[1].markers == (1,)
    assert result.verdicts[1].inherited is True
    assert result.verdicts[1].supported is True


def test_multiline_structure_preserved_on_rebuild(blocks: list[ContextBlock]) -> None:
    text = "办理要点如下：\n\n- 在读证明由教务处负责办理 [1]\n- 教务处是学籍主管部门 [1]\n"
    result = ClaimChecker(mode="fix").verify(text, blocks)
    assert result.corrections
    assert result.text.count("\n") == text.strip().count("\n")
    assert "- 教务处是学籍主管部门 [2]" in result.text


def test_short_claim_not_judged(blocks: list[ContextBlock]) -> None:
    result = ClaimChecker().verify("可以 [1]。", blocks)
    assert result.verdicts[0].reason == "too_short"
    assert result.verdicts[0].supported is True
    assert result.dropped == []


def test_drop_ratio_guard_reverts_to_fix() -> None:
    """多数句子都判不支撑 → 判定不可信 → 不删句，只留诊断。"""
    blocks = [block(1, chunk(text=GUIDE_TEXT))]
    text = (
        "学生可以申请国家奖学金 [1]。学生可以申请国家助学金 [1]。"
        "学生在读期间可以参军入伍 [1]。在读证明由教务处负责办理 [1]。"
    )
    result = ClaimChecker(mode="strict", max_drop_ratio=0.5).verify(text, blocks)
    assert result.dropped == []
    assert result.text == text
    assert any("退回只改标" in note for note in result.notes)


def test_strict_drops_only_unsupported_claim(blocks: list[ContextBlock]) -> None:
    text = "在读证明由教务处负责办理 [1]。学生可获国家奖学金 8000 元 [1]。"
    result = ClaimChecker(mode="strict").verify(text, blocks)
    assert result.dropped
    assert "国家奖学金" not in result.text
    assert "在读证明由教务处负责办理 [1]" in result.text


# --------------------------------------------------------------------------
# 与 Answerer 集成
# --------------------------------------------------------------------------
class StubLLM:
    model = "stub-claims"

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, messages, *, temperature=None, max_tokens=None):  # noqa: ANN001
        from rag_assistant.llm.base import LLMResponse

        return LLMResponse(text=self.text, model=self.model)

    def stream(self, messages, *, temperature=None, max_tokens=None):  # noqa: ANN001
        for index in range(0, len(self.text), 8):
            yield self.text[index : index + 8]


class StubRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result

    def retrieve(self, query: str, *, top_n=None, extra_queries: Sequence[str] = (), threshold=None):
        return self.result


def evidence_result() -> RetrievalResult:
    chunks = [
        chunk(text=GUIDE_TEXT),
        chunk("d2#0", "d2", "学生学籍管理规定.md", "第二章 学籍管理", POLICY_TEXT),
    ]
    return RetrievalResult(
        query="在读证明去哪里办",
        chunks=chunks,
        best_score=0.78,
        has_evidence=True,
        diagnostics={"reason": "ok", "threshold": 0.55, "best_score": 0.78, "gate": "rerank"},
    )


def answerer_settings(mock_settings: Settings, mode: str) -> Settings:
    return replace(
        mock_settings,
        generation=replace(
            mock_settings.generation,
            refuse_text=REFUSE,
            disclaimer="提示：以原文件为准。",
            claim_check=mode,
        ),
    )


def build(settings: Settings, text: str) -> Answerer:
    return Answerer(settings, StubRetriever(evidence_result()), StubLLM(text), None)


def test_answerer_fixes_citation_and_reports(mock_settings: Settings) -> None:
    text = "在读证明由教务处负责办理 [1]。教务处是学籍主管部门 [1]。"
    result = build(answerer_settings(mock_settings, "fix"), text).answer("在读证明去哪里办")

    assert result.refused is False
    # 改标后两条来源都在：办事指南 + 学籍规定
    assert len(result.citations) == 2
    assert "[2]" in result.answer
    claims = result.diagnostics["claims"]
    assert claims["corrections"]


def test_answerer_strict_refuses_when_all_claims_dropped(mock_settings: Settings) -> None:
    text = "学生可获国家奖学金 8000 元 [1]。"
    result = build(answerer_settings(mock_settings, "strict"), text).answer("在读证明去哪里办")

    assert result.refused is True
    assert result.refusal_reason == "claim_unsupported"
    assert result.answer.startswith(REFUSE)


def test_stream_done_carries_claim_report(mock_settings: Settings) -> None:
    text = "在读证明由教务处负责办理 [1]。教务处是学籍主管部门 [1]。"
    events = list(build(answerer_settings(mock_settings, "fix"), text).stream("在读证明去哪里办"))
    done = events[-1]
    assert done["type"] == "done"
    assert done["claims"]["mode"] == "fix"
    assert "[2]" in done["answer"]


def test_lead_in_line_is_structural(blocks: list[ContextBlock]) -> None:
    """材料清单前面的引导句不含结论，不参与判定（否则 strict 会把标题删掉）。"""
    text = "复学需要提交的材料：\n\n- 在读证明由教务处负责办理 [1]\n"
    result = ClaimChecker(mode="strict").verify(text, blocks)
    assert result.verdicts[0].reason == "structural"
    assert result.checked == 1  # 只核一条真正的结论
    assert result.dropped == []


def test_meta_statement_is_structural(blocks: list[ContextBlock]) -> None:
    """"资料未提及"是边界声明，不是结论；判它无依据会误删正当回答。"""
    result = ClaimChecker(mode="strict").verify("关于借书权限是否受影响，资料未提及 [1]。", blocks)
    assert result.verdicts[0].reason == "structural"
    assert result.dropped == []
    assert result.text == "关于借书权限是否受影响，资料未提及 [1]。"
