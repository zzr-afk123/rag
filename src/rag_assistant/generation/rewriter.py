"""问题改写：指代消解 + Multi-Query。

一次 LLM 调用同时产出 ``standalone``（消解后的完整问题）与 ``variants``（等价改写），
这是刻意的成本取舍：两次调用更灵活，但首字延迟会翻倍，而改写对延迟最敏感。

**所有失败都必须降级**：LLM 返回的不是 JSON、JSON 字段缺失、超时、
甚至返回中文说明——统一退化为"用原问题直接检索"。检索链路不能因为
"改写器抽风"而整体失败，最坏也要能按用户原话去查。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import Settings
from ..llm.base import LLMClient, LLMUsage
from ..llm.prompts import rewrite_prompt
from ..logging_setup import get_logger

logger = get_logger("generation.rewriter")

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_STANDALONE_FALLBACK = re.compile(r'"standalone"\s*:\s*"([^"]*)"')


@dataclass(frozen=True)
class RewriteResult:
    """改写结果。"""

    standalone: str
    variants: list[str] = field(default_factory=list)
    raw: str = ""
    degraded: bool = False
    skipped: bool = False
    usage: LLMUsage = LLMUsage()

    @property
    def queries(self) -> list[str]:
        return [self.standalone, *self.variants]

    def to_dict(self) -> dict[str, object]:
        return {
            "standalone": self.standalone,
            "variants": self.variants,
            "degraded": self.degraded,
            "skipped": self.skipped,
        }


def _extract_json(text: str) -> dict[str, object] | None:
    """从模型输出里抽 JSON 对象（容忍前后废话与代码块围栏）。"""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = _JSON_OBJECT.search(cleaned)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(text.split()).lower()
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _is_relevant(standalone: str, variant: str) -> bool:
    """等价查询至少要与原问题共享一个字符二元组，否则视为跑题丢弃。

    为什么需要这道闸：multi-query 的每一路在 RRF 里权重相同，
    一条跑偏的查询会把无关 chunk 顶进候选池，反而拉低精度。
    阈值刻意设得很松（只要共享 1 个二元组），避免误杀正常的同义改写。
    """
    return bool(_bigrams(standalone) & _bigrams(variant))


class QueryRewriter:
    """把"那它的期限呢？"变成可独立检索的问题。"""

    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    def rewrite(self, question: str, history_text: str = "") -> RewriteResult:
        """执行改写。

        Args:
            question: 用户原始问题。
            history_text: 会话历史纯文本（首轮为空串）。

        Returns:
            :class:`RewriteResult`；``degraded=True`` 表示改写失败、已回落到原问题。
        """
        question = (question or "").strip()
        if not question:
            return RewriteResult(standalone="", degraded=False, skipped=True)

        enable_multi = self._settings.retrieval.enable_multi_query
        if not enable_multi and not history_text.strip():
            # 首轮 + 关闭 multi-query：没有任何可消解的上下文，省一次调用
            return RewriteResult(standalone=question, skipped=True)

        variant_count = max(1, self._settings.retrieval.multi_query) if enable_multi else 1
        try:
            response = self._llm.complete(
                rewrite_prompt(question, history_text, variant_count),
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:  # noqa: BLE001 - 改写失败必须降级
            logger.warning("问题改写调用失败，回落为原问题：%s", exc)
            return RewriteResult(standalone=question, degraded=True)

        parsed = _extract_json(response.text)
        if parsed is None:
            fallback = _STANDALONE_FALLBACK.search(response.text or "")
            standalone = (fallback.group(1) if fallback else "").strip() or question
            logger.warning("问题改写输出不是合法 JSON，已降级：%r", (response.text or "")[:120])
            return RewriteResult(
                standalone=standalone,
                raw=response.text or "",
                degraded=True,
                usage=response.usage,
            )

        standalone = str(parsed.get("standalone") or "").strip() or question
        raw_variants = parsed.get("variants")
        variants: list[str] = []
        dropped = 0
        if isinstance(raw_variants, list):
            for item in raw_variants:
                text = str(item).strip()
                if not text or text == standalone or text in variants:
                    continue
                if not _is_relevant(standalone, text):
                    dropped += 1
                    continue
                variants.append(text)
        variants = variants[:variant_count]
        if dropped:
            logger.info("丢弃 %d 条与主查询无关的等价查询（防止稀释召回）", dropped)

        return RewriteResult(
            standalone=standalone,
            variants=variants,
            raw=response.text or "",
            degraded=False,
            usage=response.usage,
        )
