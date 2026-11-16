"""上下文组装：低分资料块过滤（retrieval.context_min_score）。"""

from __future__ import annotations

from rag_assistant.generation.context import build_context
from rag_assistant.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str, section: str, score: float, *, doc_id: str = "doc-1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_name="反诈告知书.pdf",
        doc_type="policy",
        section_path=section,
        text=f"{section}正文内容",
        rerank_score=score,
    )


def test_context_drops_blocks_below_min_score():
    """低于 context_min_score 的块不进 prompt（问反诈不该带上实验室条款）。"""
    blocks = build_context(
        [
            _chunk("a", "正文", 0.73),
            _chunk("b", "第一章", 0.51),
            _chunk("c", "第二章", 0.50),
        ],
        min_score=0.55,
    )
    assert [block.chunk.chunk_id for block in blocks] == ["a"]


def test_context_min_score_zero_keeps_all_blocks():
    """min_score=0 表示不过滤，保持旧行为。"""
    blocks = build_context(
        [_chunk("a", "正文", 0.73), _chunk("b", "第一章", 0.51)],
        min_score=0.0,
    )
    assert [block.chunk.chunk_id for block in blocks] == ["a", "b"]


def test_context_always_keeps_best_block():
    """最高分也没过线时仍保留——闸门 1 已判过"有没有依据"，这里不能把上下文清空。"""
    blocks = build_context(
        [_chunk("a", "正文", 0.40), _chunk("b", "第一章", 0.31)],
        min_score=0.55,
    )
    assert [block.chunk.chunk_id for block in blocks] == ["a"]
