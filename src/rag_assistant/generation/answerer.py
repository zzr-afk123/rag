"""回答生成 + 两道拒答闸门 + 引用溯源。

两道闸门（需求 3 的落地）：
- **闸门 1（检索层）**：``retrieval.has_evidence == False`` → 不调 LLM，直接拒答。
  省一次调用，也避免模型"硬编"。
- **闸门 2（生成层）**：模型输出里没有任何合法引用编号 → 一律改判拒答。
  这是防编造的最后一道锁：没有引用 = 没有依据 = 不应该给出结论。

为什么拒答话术与免责声明由代码负责：
- 评测要按字符串匹配拒答（"拒答准确率"），必须是常量；
- 政策类问题"以原文件为准"必须 100% 出现，不能指望模型每次都写；
- 模型如果自作主张加了套话，会在闸门 2 之后被统一格式覆盖。

**流式的妥协**（必须说明）：流式输出时，token 在生成过程中就已经发给前端了，
此时还无法判断引用是否合法。因此流式路径的闸门 2 在**流结束后**检查，
若不合格则追加一个 ``retract`` 事件，前端需要用返回的拒答文案**整段替换**气泡。
这是"首字延迟"与"零编造"之间的真实取舍，不是疏漏。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from ..config import Settings
from ..departments import DepartmentDirectory
from ..llm.base import LLMClient, LLMUsage
from ..llm.prompts import answer_prompt
from ..logging_setup import get_logger
from ..retrieval.models import RetrievalResult
from ..retrieval.retriever import HybridRetriever
from .citation import Citation, remap_citations, render_citations
from .claims import ClaimChecker
from .context import build_context
from .rewriter import QueryRewriter, RewriteResult
from .textfix import strip_source_appendix

logger = get_logger("generation.answerer")

#: 拒答时最多给几条"最相近线索"
_MAX_HINTS = 3

#: 去掉拒答话术后，正文至少还有这么多字，才算"答了但多嘴"，否则按拒答处理
_MIN_ANSWER_CHARS = 20


@dataclass
class AnswerResult:
    """一轮问答的完整结果。"""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    rewritten_query: str = ""
    extra_queries: list[str] = field(default_factory=list)
    closest_docs: list[dict[str, object]] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: LLMUsage = LLMUsage()
    diagnostics: dict[str, object] = field(default_factory=dict)
    raw_answer: str = ""
    #: 检索层最优分（阈值判定依据），评测与调参直接取用
    best_score: float = 0.0
    #: 召回的候选（含全文），供评测/调试使用；不返回给前端
    retrieved: list[dict[str, object]] = field(default_factory=list)
    #: 实际送进 prompt 的资料块（按章节合并后的正文），评测判忠实度应以它为准
    context_blocks: list[dict[str, object]] = field(default_factory=list)
    #: 命中的相关部门（含联系方式）。只做"提示"，不写进 answer 正文，
    #: 避免污染忠实度评测，也避免模型把联系方式当成自己的结论。
    contacts: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "rewritten_query": self.rewritten_query,
            "extra_queries": self.extra_queries,
            "closest_docs": self.closest_docs,
            "contacts": self.contacts,
            "latency_ms": round(self.latency_ms, 1),
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
            },
            "diagnostics": self.diagnostics,
        }


class Answerer:
    """把"问题 + 会话历史"变成"带引用的答案或拒答"。"""

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm: LLMClient,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._rewriter = rewriter
        # 逐句核对结论是否真在引用章节内（阶段 G）；模式见 generation.claim_check
        self._claims = ClaimChecker.from_settings(settings)
        # 部门目录（阶段 H）：命中关键词就把联系方式结构化返回，供前端提示
        self._departments = DepartmentDirectory.from_settings(settings)

    # ---- 公共辅助 ----
    def _refusal(self, retrieval: RetrievalResult | None, reason: str) -> str:
        """统一拒答文案 + 最相近文档线索。

        ``off_topic`` 是"问题本身不属于知识库范围"（问天气、闲聊），
        这时**不能**给"你可能想找的资料"——那些片段与问题无关，
        只会让用户以为系统在硬扯。
        """
        text = self._settings.generation.refuse_text
        if reason == "off_topic":
            return f"{text}\n\n该问题看起来不属于本知识库的范围（学院制度、办事流程、政策规定）。"
        if reason == "claim_unsupported":
            return f"{text}\n\n（本次生成的结论未能逐句在所引原文中找到依据，已按无依据处理。）"
        hints = self._hints(retrieval)
        return f"{text}\n\n{hints}" if hints else text

    def _hints(self, retrieval: RetrievalResult | None) -> str:
        if not retrieval or not retrieval.chunks:
            return ""
        lines = ["你可能想找的资料："]
        for chunk in retrieval.chunks[:_MAX_HINTS]:
            location = chunk.section_path or "正文"
            lines.append(f"- 《{chunk.doc_name}》{location}（{chunk.page_label}）")
        return "\n".join(lines)

    @staticmethod
    def _closest(retrieval: RetrievalResult | None) -> list[dict[str, object]]:
        if not retrieval:
            return []
        return [
            {
                "doc_name": chunk.doc_name,
                "section_path": chunk.section_path,
                "page": chunk.page,
                "score": round(chunk.final_score, 4),
                "chunk_id": chunk.chunk_id,
            }
            for chunk in retrieval.chunks[:_MAX_HINTS]
        ]

    @staticmethod
    def _retrieved_view(retrieval: RetrievalResult | None) -> list[dict[str, object]]:
        """完整候选视图（含全文），供评测计算 Hit@K / MRR 与忠实度判定。"""
        if not retrieval:
            return []
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "section_path": chunk.section_path,
                "page": chunk.page,
                "score": round(chunk.final_score, 6),
                "text": chunk.text,
            }
            for chunk in retrieval.chunks
        ]

    def _prepare(
        self,
        question: str,
        history_text: str,
        threshold: float | None,
    ) -> tuple[RewriteResult, RetrievalResult, LLMUsage]:
        """改写 + 检索（两个入口共用）。"""
        if self._rewriter is not None:
            rewrite = self._rewriter.rewrite(question, history_text)
        else:
            rewrite = RewriteResult(standalone=question, skipped=True)
        retrieval = self._retriever.retrieve(
            rewrite.standalone,
            extra_queries=rewrite.variants,
            threshold=threshold,
        )
        return rewrite, retrieval, rewrite.usage

    def _compose(self, body: str, citations: list[Citation]) -> str:
        """拼最终答案：正文 + 引用来源列表 + 政策类免责声明。"""
        parts = [body.strip()]
        citation_block = render_citations(citations)
        if citation_block:
            parts.append(citation_block)
        if self._should_add_disclaimer(citations):
            parts.append(self._settings.generation.disclaimer)
        return "\n\n".join(part for part in parts if part)

    def _should_add_disclaimer(self, citations: list[Citation]) -> bool:
        if not self._settings.generation.add_disclaimer:
            return False
        if not citations:
            return False
        return any((citation.doc_type or "policy") == "policy" for citation in citations)

    # ---- 同步 ----
    def answer(
        self,
        question: str,
        *,
        history_text: str = "",
        threshold: float | None = None,
    ) -> AnswerResult:
        """生成回答（阻塞直到完成）。"""
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return AnswerResult(
                question="",
                answer=self._refusal(None, "empty_question"),
                refused=True,
                refusal_reason="empty_question",
            )

        rewrite, retrieval, usage = self._prepare(question, history_text, threshold)
        base = dict(
            question=question,
            rewritten_query=rewrite.standalone,
            extra_queries=rewrite.variants,
            closest_docs=self._closest(retrieval),
            retrieved=self._retrieved_view(retrieval),
            best_score=retrieval.best_score,
            diagnostics={
                "retrieval": retrieval.diagnostics,
                "rewrite": rewrite.to_dict(),
            },
        )

        # ---- 闸门 1：检索层无依据 ----
        if not retrieval.has_evidence:
            reason = str(retrieval.diagnostics.get("reason", "no_evidence"))
            result = AnswerResult(
                answer=self._refusal(retrieval, reason),
                refused=True,
                refusal_reason=reason,
                usage=usage,
                **base,
            )
            # 被拒答时最需要"该找谁"：联系方式照给（问题命中部门即可，与是否有答案无关）
            result.contacts = self._contacts(rewrite.standalone, rewritten=question)
            result.latency_ms = (time.perf_counter() - started) * 1000
            logger.info("拒答（闸门1 %s）：question=%r 最优分=%.4f", reason, question, retrieval.best_score)
            return result

        # ---- 组装上下文 ----
        blocks = build_context(
            retrieval.chunks,
            max_chunks=self._settings.generation.max_context_chunks,
            max_tokens=self._settings.generation.max_context_tokens,
            min_score=self._settings.retrieval.context_min_score,
        )
        # 记录"模型实际看到的资料块"（按章节合并后），评测判忠实度以此为准
        base["context_blocks"] = [
            {
                "index": block.index,
                "doc_id": block.chunk.doc_id,
                "chunk_id": block.chunk.chunk_id,
                "doc_name": block.chunk.doc_name,
                "section_path": block.chunk.section_path,
                "body": block.body,
            }
            for block in blocks
        ]
        if not blocks:
            result = AnswerResult(
                answer=self._refusal(retrieval, "no_context"),
                refused=True,
                refusal_reason="no_context",
                usage=usage,
                **base,
            )
            result.contacts = self._contacts(rewrite.standalone, rewritten=question)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        # ---- 调用 LLM ----
        try:
            response = self._llm.complete(
                answer_prompt(
                    rewrite.standalone,
                    [block.render() for block in blocks],
                    refuse_text=self._settings.generation.refuse_text,
                    history_text=history_text,
                )
            )
        except Exception as exc:  # noqa: BLE001 - LLM 失败也要有可读结果
            logger.exception("生成失败，回落为拒答")
            result = AnswerResult(
                answer=f"{self._settings.generation.refuse_text}\n\n（生成服务暂时不可用：{type(exc).__name__}）",
                refused=True,
                refusal_reason="llm_error",
                usage=usage,
                **base,
            )
            result.contacts = self._contacts(rewrite.standalone, rewritten=question)
            result.latency_ms = (time.perf_counter() - started) * 1000
            return result

        usage = usage + response.usage
        raw = strip_source_appendix((response.text or "").strip())
        base["diagnostics"]["llm"] = {
            "finish_reason": response.finish_reason,
            "reasoning_chars": len(response.reasoning),
        }

        # 思考型模型常见故障：只往 reasoning_content 里"想"，content 为空。
        # 这里必须当成空答案（走拒答），绝不能把思考草稿当回答返回。
        if not raw and response.reasoning:
            logger.warning(
                "模型只输出思考内容（%d 字）而没有答案，按空答案处理并拒答；"
                "多半是思考型模型（reasoning_content）。建议改用 instruct 模型或关闭思考。",
                len(response.reasoning),
            )

        # ---- 模型自己选择拒答 ----
        if not raw:
            return self._finish(
                AnswerResult(
                    answer=self._refusal(retrieval, "empty_answer"),
                    refused=True,
                    refusal_reason="empty_answer",
                    usage=usage,
                    **base,
                ),
                started,
                question,
            )
        raw, refused_by_model = self._split_refusal(raw)
        if refused_by_model:
            return self._finish(
                AnswerResult(
                    answer=self._refusal(retrieval, "llm_refused"),
                    refused=True,
                    refusal_reason="llm_refused",
                    usage=usage,
                    raw_answer=raw,
                    **base,
                ),
                started,
                question,
            )

        # ---- 闸门 2.5：claim 级引用校验 ----
        # 闸门 2 只看"有没有合法编号"，拦不住"编号对、内容错位"。这里逐句比对
        # 结论与它所引章节的原文，先尝试改标，救不回来才降级（strict 下删句）。
        claim_check = self._claims.verify(raw, blocks)
        base["diagnostics"]["claims"] = claim_check.to_dict()
        if claim_check.changed:
            logger.info(
                "claim 校验：改标 %d 句、删句 %d 句（逐句支持率 %.2f）",
                len(claim_check.corrections),
                len(claim_check.dropped),
                claim_check.support_rate,
            )
        checked_text = claim_check.text if claim_check.enabled else raw
        if not checked_text.strip():
            logger.warning("claim 校验后正文为空，整题改判拒答")
            return self._finish(
                AnswerResult(
                    answer=self._refusal(retrieval, "claim_unsupported"),
                    refused=True,
                    refusal_reason="claim_unsupported",
                    usage=usage,
                    raw_answer=raw,
                    **base,
                ),
                started,
                question,
            )

        # ---- 闸门 2：引用校验 ----
        remap = remap_citations(checked_text, blocks, quote_chars=self._settings.generation.quote_chars)
        if self._settings.generation.require_citation and not remap.has_citation:
            logger.warning("引用校验失败（闸门2）：无合法引用，改为拒答。原文=%r", raw[:200])
            return self._finish(
                AnswerResult(
                    answer=self._refusal(retrieval, "citation_invalid"),
                    refused=True,
                    refusal_reason="citation_invalid",
                    usage=usage,
                    raw_answer=raw,
                    **base,
                ),
                started,
                question,
            )

        result = AnswerResult(
            answer=self._compose(remap.text, remap.citations),
            citations=remap.citations,
            refused=False,
            usage=usage,
            raw_answer=raw,
            **base,
        )
        result.diagnostics = {**result.diagnostics, "invalid_citations": remap.invalid_indices}
        return self._finish(result, started, question)

    def _contacts(
        self,
        question: str,
        *,
        rewritten: str = "",
        doc_names: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """问题 / 改写 / 引用文档名 → 建议部门（含联系方式）。"""
        try:
            return self._departments.contacts_for(
                question,
                rewritten=rewritten,
                doc_names=doc_names or [],
            )
        except Exception:  # noqa: BLE001 - 联系方式是锦上添花，绝不能拖垮问答
            logger.warning("部门匹配失败（已忽略）", exc_info=True)
            return []

    def _split_refusal(self, text: str) -> tuple[str, bool]:
        """把"答了一半又说没依据"拆开。

        模型常见的半吊子行为：先给出两三条有出处的结论，末尾再补一句
        "当前知识库中未找到相关依据……"。这句话是拒答话术，不该留在正文里，
        也不该把整条回答判成拒答。
        """
        refuse = self._settings.generation.refuse_text
        if not refuse or refuse not in text:
            return text, False
        body = text.replace(refuse, "")
        body = re.sub(r"[ \t]*\n{3,}", "\n\n", body).strip(" \t\r\n，,。；;")
        if len(body) >= _MIN_ANSWER_CHARS:
            return body, False
        return text, True

    def _finish(self, result: AnswerResult, started: float, question: str) -> AnswerResult:
        if not result.contacts:
            result.contacts = self._contacts(
                result.question,
                rewritten=result.rewritten_query,
                doc_names=[citation.doc_name for citation in result.citations],
            )
        result.latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "回答完成：question=%r 引用=%d 拒答=%s 耗时=%.0fms tokens=%d",
            question,
            len(result.citations),
            result.refused,
            result.latency_ms,
            result.usage.total_tokens,
        )
        return result

    # ---- 流式 ----
    def stream(
        self,
        question: str,
        *,
        history_text: str = "",
        threshold: float | None = None,
    ) -> Iterator[dict[str, object]]:
        """流式生成。

        事件类型：
        - ``meta``：改写结果与检索诊断（先发，前端可先展示"正在检索"）
        - ``token``：文本增量
        - ``done``：最终文本（**已重编号**，前端应以它为准覆盖）、引用列表、耗时
        - ``retract``：流结束后发现引用不合法，前端需用 ``answer`` 整段替换气泡
        """
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            yield {"type": "done", "answer": self._refusal(None, "empty_question"), "refused": True, "refusal_reason": "empty_question", "citations": []}
            return

        rewrite, retrieval, usage = self._prepare(question, history_text, threshold)
        yield {
            "type": "meta",
            "question": question,
            "rewritten_query": rewrite.standalone,
            "extra_queries": rewrite.variants,
            "retrieval": retrieval.diagnostics,
            "retrieved_ids": [chunk.chunk_id for chunk in retrieval.chunks],
            "closest_docs": self._closest(retrieval),
        }

        if not retrieval.has_evidence:
            reason = str(retrieval.diagnostics.get("reason", "no_evidence"))
            yield {
                "type": "done",
                "answer": self._refusal(retrieval, reason),
                "refused": True,
                "refusal_reason": reason,
                "citations": [],
                "contacts": self._contacts(rewrite.standalone, rewritten=rewrite.standalone),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            return

        blocks = build_context(
            retrieval.chunks,
            max_chunks=self._settings.generation.max_context_chunks,
            max_tokens=self._settings.generation.max_context_tokens,
            min_score=self._settings.retrieval.context_min_score,
        )
        messages = answer_prompt(
            rewrite.standalone,
            [block.render() for block in blocks],
            refuse_text=self._settings.generation.refuse_text,
            history_text=history_text,
        )

        chunks: list[str] = []
        try:
            for piece in self._llm.stream(messages):
                if not piece:
                    continue
                chunks.append(piece)
                yield {"type": "token", "text": piece}
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式生成中断")
            yield {
                "type": "done",
                "answer": f"{self._settings.generation.refuse_text}\n\n（生成服务中断：{type(exc).__name__}）",
                "refused": True,
                "refusal_reason": "llm_error",
                "citations": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            return

        raw = strip_source_appendix("".join(chunks).strip())
        raw, refused_by_model = self._split_refusal(raw)
        claim_check = self._claims.verify(raw, blocks)
        checked_text = claim_check.text if claim_check.enabled else raw
        if not checked_text.strip():
            logger.warning("流式 claim 校验后正文为空，要求前端替换为拒答")
            yield {
                "type": "retract",
                "reason": "claim_unsupported",
                "answer": self._refusal(retrieval, "claim_unsupported"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            return
        remap = remap_citations(checked_text, blocks, quote_chars=self._settings.generation.quote_chars)
        latency = round((time.perf_counter() - started) * 1000, 1)

        if self._settings.generation.require_citation and not remap.has_citation:
            logger.warning("流式引用校验失败（闸门2），要求前端替换为拒答")
            yield {
                "type": "retract",
                "reason": "citation_invalid",
                "answer": self._refusal(retrieval, "citation_invalid"),
                "latency_ms": latency,
            }
            return

        yield {
            "type": "done",
            "answer": self._compose(remap.text, remap.citations),
            "refused": refused_by_model,
            "refusal_reason": "llm_refused" if refused_by_model else None,
            "citations": [citation.to_dict() for citation in remap.citations],
            "contacts": self._contacts(
                question,
                rewritten=rewrite.standalone,
                doc_names=[citation.doc_name for citation in remap.citations],
            ),
            "claims": claim_check.to_dict(),
            "latency_ms": latency,
            "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
            "note": "流式路径的 usage 不含生成阶段 token（客户端流式接口不回传用量）",
        }
