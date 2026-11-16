"""引用溯源测试：编号校验、按章节聚合、重编号、摘录。"""

from __future__ import annotations

from rag_assistant.generation.citation import (
    Citation,
    _snippet,
    extract_markers,
    remap_citations,
    render_citations,
)
from rag_assistant.generation.context import ContextBlock, build_context
from rag_assistant.retrieval.models import RetrievedChunk


def chunk(
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
        vector_score=0.3,
        bm25_score=2.0,
        rrf_score=0.03,
        rerank_score=0.9,
    )


def block(index: int, item: RetrievedChunk) -> ContextBlock:
    header = f"[{index}] 文档：{item.doc_name} ｜ 章节：{item.section_path} ｜ 页码：无页码"
    return ContextBlock(index=index, chunk=item, header=header, body=item.text)


def test_extract_markers_order() -> None:
    assert extract_markers("结论A[2]，结论B[1]，再引用[2]。") == [2, 1, 2]
    assert extract_markers("没有引用") == []
    assert extract_markers("[12]") == [12]


def test_remap_basic() -> None:
    blocks = [block(1, chunk())]
    result = remap_citations("学生可休学一年[1]。", blocks)
    assert result.text == "学生可休学一年[1]。"
    assert len(result.citations) == 1
    assert result.citations[0].index == 1
    assert result.citations[0].doc_name == "学生学籍管理规定.md"
    assert result.invalid_indices == []


def test_remap_groups_same_section() -> None:
    """同一章节的两个 chunk 共用一个编号（方案假设 #15）。"""
    blocks = [
        block(1, chunk("d1#0")),
        block(2, chunk("d1#1")),  # 同 doc 同 section
    ]
    result = remap_citations("休学一年[1]，累计不超过两年[2]。", blocks)
    assert len(result.citations) == 1
    assert result.text == "休学一年[1]，累计不超过两年[1]。"


def test_remap_collapses_adjacent_duplicates() -> None:
    """连续标注（[1][2] 指向同一章节）折叠成一个编号，读起来才干净。"""
    blocks = [block(1, chunk("d1#0")), block(2, chunk("d1#1"))]
    result = remap_citations("休学一年[1][2]。", blocks)
    assert result.text == "休学一年[1]。"
    assert len(result.citations) == 1


def test_remap_orders_by_first_appearance() -> None:
    blocks = [
        block(1, chunk("d1#0", doc_id="d1", section="第一章")),
        block(2, chunk("d2#0", doc_id="d2", doc_name="报销办法.docx", section="第二章")),
    ]
    result = remap_citations("先引第二条[2]，再引第一条[1]。", blocks)
    assert [c.index for c in result.citations] == [1, 2]
    assert result.citations[0].doc_name == "报销办法.docx"  # 首次出现的是 [2]
    assert result.text == "先引第二条[1]，再引第一条[2]。"


def test_remap_invalid_index_dropped() -> None:
    """越界编号要被抹掉并记录（闸门 2 的判据之一）。"""
    blocks = [block(1, chunk())]
    result = remap_citations("甲[1]，乙[9]。", blocks)
    assert result.invalid_indices == [9]
    assert "[9]" not in result.text
    assert len(result.citations) == 1


def test_remap_no_citation() -> None:
    result = remap_citations("学生可以休学。", [block(1, chunk())])
    assert result.has_citation is False
    assert result.text == "学生可以休学。"


def test_remap_empty_input() -> None:
    result = remap_citations("", [])
    assert result.text == ""
    assert result.citations == []


def test_citation_quote_is_sentence_aligned() -> None:
    citation = remap_citations("x[1]", [block(1, chunk())], quote_chars=20).citations[0]
    assert citation.quote.endswith("。")
    assert len(citation.quote) <= 21


def test_snippet_truncates_long_text() -> None:
    text = "这是一句很长的制度文本，" * 30
    snippet = _snippet(text, 40)
    assert len(snippet) <= 41
    assert snippet.endswith(("…", "，")) or snippet.endswith("。")


def test_citation_render_and_dict() -> None:
    citation = Citation(
        index=1,
        doc_id="d1",
        doc_name="学生学籍管理规定.md",
        section_path="第三章 > 第七条",
        page=12,
        chunk_id="d1#0",
        quote="休学期一般以一年为期。",
    )
    line = citation.render_line()
    assert line.startswith("[1] 《学生学籍管理规定.md》第三章 > 第七条（第 12 页）")
    assert citation.page_label == "第 12 页"
    assert citation.to_dict()["page"] == 12


def test_render_citations_block() -> None:
    citation = Citation(index=1, doc_id="d1", doc_name="a.md", section_path="第一章", page=0, chunk_id="c", quote="原文")
    text = render_citations([citation])
    # 阶段 H：去掉 Markdown 星号，改成纯文本标题（前端用 .sources h4 上样式）
    assert text.startswith("引用来源")
    assert "**" not in text
    assert "a.md" in text and "无页码" in text
    assert render_citations([]) == ""


def test_build_context_numbering_and_budget() -> None:
    chunks = [chunk(f"d1#{i}", section=f"第{i}条") for i in range(4)]
    blocks = build_context(chunks, max_chunks=2, max_tokens=1000)
    assert [b.index for b in blocks] == [1, 2]
    assert blocks[0].header.startswith("[1] 文档：")

    tight = build_context(chunks, max_chunks=4, max_tokens=60)
    assert len(tight) <= 2  # 预算小到只能塞下一条左右
    assert all(b.body for b in tight)
