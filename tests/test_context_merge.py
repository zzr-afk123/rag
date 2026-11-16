"""上下文按章节合并的测试（修复"综合两段作答却只引一段"）。"""

from __future__ import annotations

from rag_assistant.generation.citation import remap_citations
from rag_assistant.generation.context import build_context
from rag_assistant.retrieval.models import RetrievedChunk


def make_chunk(
    chunk_id: str,
    section: str,
    text: str,
    *,
    score: float = 0.7,
    doc_id: str = "d1",
    doc_name: str = "科研经费报销管理办法.docx",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type="policy",
        section_path=section,
        text=text,
        page=0,
        chunk_type="text",
        rerank_score=score,
    )


def test_same_section_chunks_are_merged() -> None:
    """同章节的正文段与表格必须合并成一个资料块、一个引用编号。"""
    chunks = [
        make_chunk("a#1", "第六条 缓交学费", "经学生工作处审核同意后可先行注册，缓交期限不超过一学期。"),
        make_chunk("a#2", "第六条 缓交学费", "| 缓交学费申请 | 学生工作处 | 5 个工作日 | 申请表 |", score=0.6),
        make_chunk("b#1", "第七条 休学条件", "休学期一般以一年为期。", score=0.5),
    ]
    blocks = build_context(chunks, max_chunks=5, max_tokens=3000)

    assert len(blocks) == 2, "同章节的两条 chunk 应合并"
    assert blocks[0].chunk.chunk_id == "a#1", "代表 chunk 取该章节相关度最高的一条"
    assert "缓交期限" in blocks[0].body and "5 个工作日" in blocks[0].body
    assert blocks[1].index == 2


def test_merged_body_is_used_for_citation_quote() -> None:
    """引用摘录必须来自合并后的正文，而不是单条 chunk 的片段。"""
    chunks = [
        make_chunk("a#1", "第六条 缓交学费", "缓交期限不超过一学期。"),
        make_chunk("a#2", "第六条 缓交学费", "5 个工作日内由学生工作处审核。", score=0.6),
    ]
    blocks = build_context(chunks, max_chunks=5, max_tokens=3000)
    result = remap_citations("结论[1]。", blocks, quote_chars=200)
    quote = result.citations[0].quote
    assert "缓交期限" in quote and "5 个工作日" in quote


def test_token_budget_still_respected() -> None:
    """合并后仍受 token 预算约束。"""
    long_text = "休学期一般以一年为期，累计不得超过两年。" * 40
    blocks = build_context([make_chunk("a#1", "第七条", long_text)], max_chunks=5, max_tokens=100)
    assert blocks
    assert blocks[0].tokens <= 120
