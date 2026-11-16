"""生成层：问题改写、上下文构建、引用溯源、回答与拒答。"""

from __future__ import annotations

from .answerer import Answerer, AnswerResult
from .citation import Citation, RemapResult, extract_markers, remap_citations, render_citations
from .claims import ClaimCheckResult, ClaimChecker, ClaimVerdict, verify_citations
from .context import ContextBlock, build_context, render_context
from .rewriter import QueryRewriter, RewriteResult

__all__ = [
    "AnswerResult",
    "Answerer",
    "Citation",
    "ClaimCheckResult",
    "ClaimChecker",
    "ClaimVerdict",
    "ContextBlock",
    "QueryRewriter",
    "RemapResult",
    "RewriteResult",
    "build_context",
    "extract_markers",
    "remap_citations",
    "render_citations",
    "render_context",
    "verify_citations",
]
