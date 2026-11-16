"""claim 级引用校验：逐句核对结论是否真能在**它所引的那个章节**里找到。

为什么需要这一层（阶段 G）：

闸门 2 只检查"答案里有没有合法的 [n]"。编号合法、章节也存在，但**句子与所引
章节无关**时它拦不住。实测第 9 题「在读证明去哪里办」就是这样：回答补了一句
"教务处是学籍主管部门"（出自《学生学籍管理规定》），编号却标在办事指南上 ——
编号对、内容错位，LLM 评审判"不忠实"，占比 5.6%。

做法（纯规则，零额外 LLM 调用）：

1. **切句**：按换行与句末标点把答案切成 claim，每个 claim 带上自己的 ``[n]``。
   句内没有编号时**继承上一句**的编号（模型习惯在要点末尾统一标注），
   整段都没有时退化到"答案里出现过的全部编号"。
2. **词面覆盖**：claim 的内容词有多大比例出现在它所引章节的正文里。
   中文用 jieba 分词，**复用 BM25 的自定义词典与停用词**，保证与检索同口径。
3. **数字硬校验**：claim 里的阿拉伯数字（金额、天数、编号）必须出现在所引
   章节原文中。覆盖率高但数字对不上的（"5000 元"被写成"8000 元"）直接判
   不支撑 —— 这是最危险的一类编造，单靠词面覆盖可能漏掉。
4. **改标优先**：不达标时先在**其它资料块**里找覆盖最高的那块，若达标且明显
   高于原引用（``claim_reassign_margin``）就改标而不是删句。第 9 题正是这一类：
   句子有依据，只是标错了章节。
5. **降级到拒答**：``strict`` 模式下，改标也救不回来的句子被删除；如果正文被
   删空则整题改判拒答（原因 ``claim_unsupported``）。

四种模式（``generation.claim_check``）：

- ``off``：完全跳过，零开销；
- ``warn``：只记录诊断，不改答案（灰度观察用）；
- ``fix``：改标（**默认**）——修引用错标，不删句子；
- ``strict``：改标 + 删除无依据的句子（有 ``claim_max_drop_ratio`` 兜底，
  删得太多就退回 ``fix``，避免把整段答案削没）。

为什么默认是 ``fix`` 而不是 ``strict``：词面覆盖只是**必要条件**，不是充分条件。
改写型答案（"提交三份材料" vs 原文"材料一式三份"）会低估覆盖，一上来就删句
容易误伤正常回答。先用零风险的动作（改标）把高置信度的错标修掉，删不删句
交给评测数据决定。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..logging_setup import get_logger
from .citation import CITATION_RE, extract_markers
from .context import ContextBlock

logger = get_logger("generation.claims")

#: 句末标点 + 换行都算 claim 边界
_SEP_CHARS = frozenset("。！？；!?;\n")

_BULLET = re.compile(r"^[\s\-*•·]+")
_LIST_NO = re.compile(r"^\d{1,2}\s*[.、)]\s+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_FULLWIDTH = str.maketrans("０１２３４５６７８９．", "0123456789.")

#: 引导句（"复学需要提交的材料："）——后面才是结论，本身不含事实
_LEAD_IN = re.compile(r"[：:]$")
#: 元陈述（"资料未提及"）——它不是结论，而是在声明资料边界
_META_STATEMENT = re.compile(
    r"(资料|原文|材料|知识库|文档)[^。；！？]{0,8}(未|没有|无)(提及|提到|写明|说明|规定|涉及|相关)"
)

#: 整行只有引用编号（"[1]"、"[1] [2]"）——重排编号后需要并回上一行
_MARKERS_ONLY = re.compile(r"(?:\[\d{1,2}\]\s*)+")

#: claim 里最多报几条不支撑（诊断体量控制）
_MAX_REPORT = 10


def tidy_marker_lines(text: str) -> str:
    """把"整行只有 [n]"的标记并回上一行末尾。

    4B 级模型常把编号单独写成一行，渲染出来就是一行孤零零的 "[1]"。
    """
    lines: list[str] = []
    for line in (text or "").split("\n"):
        if _MARKERS_ONLY.fullmatch(line.strip()) and lines:
            lines[-1] = lines[-1].rstrip() + " " + line.strip()
        else:
            lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 分词（复用 BM25 词典，失败时退化为字符二元组）
# --------------------------------------------------------------------------
def _fallback_tokens(text: str) -> list[str]:
    """没有 jieba 时的退化分词：ASCII 词 + 中文二元组。"""
    tokens: list[str] = []
    for word in _WORD.findall(text.lower()):
        if word[0].isascii():
            tokens.append(word)
        elif len(word) == 1:
            tokens.append(word)
        else:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


class Lexicon:
    """claim 校验用的分词器（优先 jieba + 项目词典，失败自动退化）。"""

    def __init__(self, *, userdict_path: Path | None = None, stopwords_path: Path | None = None) -> None:
        self._tokenizer = None
        try:
            from ..retrieval.bm25 import ChineseTokenizer

            self._tokenizer = ChineseTokenizer(
                userdict_path=Path(userdict_path) if userdict_path else None,
                stopwords_path=Path(stopwords_path) if stopwords_path else None,
            )
        except Exception as exc:  # noqa: BLE001 - 缺 jieba/词典损坏都不该阻断回答
            logger.warning("claim 校验分词器初始化失败，退化为字符二元组：%s", exc)

    def tokens(self, text: str) -> list[str]:
        if not text:
            return []
        if self._tokenizer is not None:
            return self._tokenizer.tokenize(text)
        return _fallback_tokens(text)


_LEXICONS: dict[tuple[str, str], Lexicon] = {}


def get_lexicon(*, userdict_path: Path | None = None, stopwords_path: Path | None = None) -> Lexicon:
    key = (str(userdict_path or ""), str(stopwords_path or ""))
    if key not in _LEXICONS:
        _LEXICONS[key] = Lexicon(userdict_path=userdict_path, stopwords_path=stopwords_path)
    return _LEXICONS[key]


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClaimVerdict:
    """单句核对结果。"""

    text: str
    #: 该句（继承后）的引用编号，指向资料块编号
    markers: tuple[int, ...]
    coverage: float
    supported: bool
    reason: str = ""
    #: 改标目标（若发生改标）
    corrected: tuple[int, ...] | None = None
    best_marker: int | None = None
    best_coverage: float = 0.0
    inherited: bool = False
    dropped: bool = False
    missing_numbers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "claim": self.text,
            "markers": list(self.markers),
            "coverage": round(self.coverage, 3),
            "supported": self.supported,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.corrected is not None:
            payload["corrected"] = list(self.corrected)
            payload["best_coverage"] = round(self.best_coverage, 3)
        if self.inherited:
            payload["inherited"] = True
        if self.dropped:
            payload["dropped"] = True
        if self.missing_numbers:
            payload["missing_numbers"] = list(self.missing_numbers)
        return payload


@dataclass(frozen=True)
class ClaimCheckResult:
    """整段答案的核对结果。"""

    text: str
    mode: str = "fix"
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    checked: int = 0
    supported: int = 0
    corrections: list[ClaimVerdict] = field(default_factory=list)
    dropped: list[ClaimVerdict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and not self.skipped_reason

    @property
    def changed(self) -> bool:
        return bool(self.corrections or self.dropped)

    @property
    def support_rate(self) -> float:
        return self.supported / self.checked if self.checked else 1.0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode,
            "checked": self.checked,
            "supported": self.supported,
            "support_rate": round(self.support_rate, 3),
            "corrections": [item.to_dict() for item in self.corrections],
            "dropped": [item.to_dict() for item in self.dropped],
            "unsupported": [
                item.to_dict()
                for item in self.verdicts
                if not item.supported and item.reason not in ("too_short",)
            ][:_MAX_REPORT],
        }
        if self.skipped_reason:
            payload["skipped"] = self.skipped_reason
        if self.notes:
            payload["notes"] = self.notes
        return payload


# --------------------------------------------------------------------------
# 切句
# --------------------------------------------------------------------------
@dataclass
class _Segment:
    """一个原始句子片段（含句末标点/换行）。"""

    raw: str
    body: str
    sep: str
    markers: list[int]
    clean: str

    @property
    def has_content(self) -> bool:
        return bool(self.clean)


def _split_segments(text: str) -> list[_Segment]:
    """按句末标点与换行切段，标点/换行留在片段尾部，便于原样拼回。"""
    raw_parts: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] in _SEP_CHARS:
            end = index
            while end < len(text) and text[end] in _SEP_CHARS:
                end += 1
            raw_parts.append(text[start:end])
            start = end
            index = end
        else:
            index += 1
    if start < len(text):
        raw_parts.append(text[start:])

    segments: list[_Segment] = []
    for raw in raw_parts:
        body = raw.rstrip("。！？；!?;\n")
        sep = raw[len(body) :]
        markers = extract_markers(body)
        clean = CITATION_RE.sub("", body)
        clean = _LIST_NO.sub("", _BULLET.sub("", clean.strip())).strip()
        segments.append(_Segment(raw=raw, body=body, sep=sep, markers=markers, clean=clean))
    return segments


def _normalize_text(text: str) -> str:
    return (text or "").translate(_FULLWIDTH)


def _numbers(text: str) -> set[str]:
    """抽取阿拉伯数字并归一（"5000.0" → "5000"）。"""
    values: set[str] = set()
    for match in _NUMBER.finditer(_normalize_text(text)):
        raw = match.group()
        try:
            number = float(raw)
        except ValueError:  # pragma: no cover - 正则已保证可解析
            values.add(raw)
            continue
        values.add(str(int(number)) if number.is_integer() else f"{number:g}")
    return values


def _coverage(tokens: set[str], reference: set[str]) -> float:
    if not tokens:
        return 1.0
    return len(tokens & reference) / len(tokens)


# --------------------------------------------------------------------------
# 校验器
# --------------------------------------------------------------------------
_MODE_ALIASES = {
    "off": "off",
    "none": "off",
    "warn": "warn",
    "fix": "fix",
    "reassign": "fix",
    "strict": "strict",
}


class ClaimChecker:
    """逐句核对答案是否在其引用章节内找得到依据。"""

    def __init__(
        self,
        *,
        mode: str = "fix",
        min_coverage: float = 0.6,
        reassign_margin: float = 0.15,
        min_tokens: int = 3,
        max_drop_ratio: float = 0.5,
        number_guard: bool = True,
        userdict_path: Path | None = None,
        stopwords_path: Path | None = None,
        lexicon: Lexicon | None = None,
    ) -> None:
        normalized = _MODE_ALIASES.get((mode or "fix").strip().lower())
        if normalized is None:
            logger.warning("未知的 claim_check 模式 %r，按 fix 处理", mode)
            normalized = "fix"
        self.mode = normalized
        self.min_coverage = max(0.0, min(1.0, float(min_coverage)))
        self.reassign_margin = max(0.0, float(reassign_margin))
        self.min_tokens = max(1, int(min_tokens))
        self.max_drop_ratio = max(0.0, min(1.0, float(max_drop_ratio)))
        self.number_guard = bool(number_guard)
        self._lexicon = lexicon or get_lexicon(
            userdict_path=userdict_path, stopwords_path=stopwords_path
        )

    @classmethod
    def from_settings(cls, settings) -> "ClaimChecker":
        generation = settings.generation
        retrieval = getattr(settings, "retrieval", None)
        root = Path(getattr(settings, "root", "") or "")
        userdict = getattr(retrieval, "userdict_path", "") if retrieval else ""
        stopwords = getattr(retrieval, "stopwords_path", "") if retrieval else ""
        paths: dict[str, Path | None] = {}
        for name, value in (("userdict_path", userdict), ("stopwords_path", stopwords)):
            if not value:
                paths[name] = None
            else:
                candidate = Path(value)
                if not candidate.is_absolute() and str(root):
                    candidate = root / candidate
                paths[name] = candidate if candidate.exists() else None
        return cls(
            mode=getattr(generation, "claim_check", "fix"),
            min_coverage=getattr(generation, "claim_min_coverage", 0.6),
            reassign_margin=getattr(generation, "claim_reassign_margin", 0.15),
            min_tokens=getattr(generation, "claim_min_tokens", 3),
            max_drop_ratio=getattr(generation, "claim_max_drop_ratio", 0.5),
            number_guard=getattr(generation, "claim_number_guard", True),
            **paths,
        )

    # ---- 主流程 ----
    def verify(self, text: str, blocks: Sequence[ContextBlock]) -> ClaimCheckResult:
        """核对并（按模式）修复 ``text`` 中的引用编号。"""
        original = text or ""
        if self.mode == "off":
            return ClaimCheckResult(text=original, mode=self.mode, skipped_reason="mode_off")
        usable = [block for block in blocks if (block.body or "").strip()]
        if not original.strip() or not usable:
            return ClaimCheckResult(text=original, mode=self.mode, skipped_reason="no_context")
        if not extract_markers(original):
            return ClaimCheckResult(text=original, mode=self.mode, skipped_reason="no_markers")

        block_tokens = {block.index: set(self._lexicon.tokens(block.body)) for block in usable}
        block_numbers = {
            block.index: _numbers(block.body) for block in usable
        }

        segments = _split_segments(original)
        claims = self._assign_markers(segments)
        verdicts: list[ClaimVerdict] = []
        outcome_markers: dict[int, tuple[int, ...] | None] = {}
        outcome_drop: set[int] = set()

        for segment_index, markers, inherited in claims:
            segment = segments[segment_index]
            verdict = self._judge(
                segment,
                markers,
                inherited,
                block_tokens=block_tokens,
                block_numbers=block_numbers,
            )
            verdicts.append(verdict)
            if verdict.corrected is not None:
                outcome_markers[segment_index] = verdict.corrected
            if verdict.dropped:
                outcome_drop.add(segment_index)

        corrections = [item for item in verdicts if item.corrected is not None]
        dropped = [item for item in verdicts if item.dropped]
        notes: list[str] = []

        # 删句兜底：一次删掉大多说明判定不可信，整体退回"只改标"；
        # 但如果**一句都没剩下**（全都是无依据结论），就不能退 —— 那正是该拒答的情形。
        judged = [item for item in verdicts if item.reason not in ("too_short", "structural")]
        if (
            dropped
            and judged
            and len(dropped) < len(judged)
            and len(dropped) / len(judged) > self.max_drop_ratio
        ):
            notes.append(
                f"无依据句子占比 {len(dropped)}/{len(judged)} 超过上限 "
                f"{self.max_drop_ratio:.2f}，已退回只改标（不删句）"
            )
            dropped = []
            outcome_drop.clear()  # 保留改标结果，仅取消删句

        result_text = original
        if self.mode == "warn":
            result_text = original
            corrections_for_text: list[ClaimVerdict] = []
        else:
            corrections_for_text = corrections
        if corrections_for_text or outcome_drop:
            result_text = self._rebuild(segments, outcome_markers, outcome_drop)
            # 句子全被删掉时只剩标点/空白：当成空答案，交给调用方改判拒答
            if not re.search(r"[\w\u3400-\u4dbf\u4e00-\u9fff]", result_text):
                result_text = ""
        else:
            result_text = tidy_marker_lines(result_text)

        supported = sum(1 for item in judged if item.supported)
        checked_count = len(judged)
        return ClaimCheckResult(
            text=result_text,
            mode=self.mode,
            verdicts=verdicts,
            checked=checked_count,
            supported=supported,
            corrections=corrections,
            dropped=dropped,
            notes=notes,
        )

    # ---- 内部分步 ----
    def _assign_markers(self, segments: Sequence[_Segment]) -> list[tuple[int, tuple[int, ...], bool]]:
        """给每个有内容的句子分配引用编号（无编号时继承上一句 / 下一句 / 全篇）。"""
        content = [i for i, segment in enumerate(segments) if segment.has_content]
        if not content:
            return []
        whole = tuple(sorted(set(extract_markers("".join(segment.raw for segment in segments)))))

        assigned: list[tuple[int, tuple[int, ...], bool]] = []
        previous: tuple[int, ...] | None = None
        for position, segment_index in enumerate(content):
            segment = segments[segment_index]
            if segment.markers:
                markers = tuple(segment.markers)
                inherited = False
                previous = markers
            else:
                lookahead = None
                for later in content[position + 1 :]:
                    if segments[later].markers:
                        lookahead = tuple(segments[later].markers)
                        break
                if previous:
                    markers, inherited = previous, True
                elif lookahead:
                    markers, inherited = lookahead, True
                else:
                    markers, inherited = whole, True
            assigned.append((segment_index, markers, inherited))
        return assigned

    def _judge(
        self,
        segment: _Segment,
        markers: tuple[int, ...],
        inherited: bool,
        *,
        block_tokens: dict[int, set[str]],
        block_numbers: dict[int, set[str]],
    ) -> ClaimVerdict:
        tokens = set(self._lexicon.tokens(segment.clean))
        numbers = _numbers(segment.clean)
        cited = tuple(number for number in markers if number in block_tokens)

        base = dict(
            text=segment.clean,
            markers=markers,
            inherited=inherited,
        )

        # 引导句（"复学需要提交的材料："）与元陈述（"资料未提及"）不含结论，
        # 不该被当成"无依据"——它们在 fix 模式不会出事，但会刷屏诊断，
        # 也让 strict 模式误删段落标题。
        if _LEAD_IN.search(segment.clean) or _META_STATEMENT.search(segment.clean):
            return ClaimVerdict(coverage=1.0, supported=True, reason="structural", **base)

        # 句子太短（"可以。""办理流程如下："）不做判定，避免把套话当结论
        if len(tokens) < self.min_tokens:
            return ClaimVerdict(coverage=1.0, supported=True, reason="too_short", **base)

        if not cited:
            verdict = ClaimVerdict(coverage=0.0, supported=False, reason="no_valid_citation", **base)
            return self._maybe_reassign(verdict, tokens, numbers, block_tokens, block_numbers)

        reference: set[str] = set()
        reference_numbers: set[str] = set()
        for number in cited:
            reference |= block_tokens[number]
            reference_numbers |= block_numbers[number]
        coverage = _coverage(tokens, reference)
        missing = numbers - reference_numbers if self.number_guard else set()

        supported = coverage >= self.min_coverage and not missing
        reason = "" if supported else ("missing_number" if missing else "low_coverage")
        verdict = ClaimVerdict(
            coverage=coverage,
            supported=supported,
            reason=reason,
            missing_numbers=tuple(sorted(missing)),
            **base,
        )
        if supported:
            return verdict
        return self._maybe_reassign(verdict, tokens, numbers, block_tokens, block_numbers)

    def _maybe_reassign(
        self,
        verdict: ClaimVerdict,
        tokens: set[str],
        numbers: set[str],
        block_tokens: dict[int, set[str]],
        block_numbers: dict[int, set[str]],
    ) -> ClaimVerdict:
        """在其它资料块里找覆盖最高的那块，达标且明显更优才改标。"""
        best_index: int | None = None
        best_coverage = 0.0
        for number, reference in block_tokens.items():
            if number in verdict.markers:
                continue
            if self.number_guard and (numbers - block_numbers.get(number, set())):
                continue
            coverage = _coverage(tokens, reference)
            if coverage > best_coverage:
                best_index, best_coverage = number, coverage

        improved = (
            best_index is not None
            and best_coverage >= self.min_coverage
            and best_coverage >= verdict.coverage + self.reassign_margin
        )
        if not improved:
            return ClaimVerdict(
                text=verdict.text,
                markers=verdict.markers,
                coverage=verdict.coverage,
                supported=False,
                reason=verdict.reason,
                best_marker=best_index,
                best_coverage=best_coverage,
                inherited=verdict.inherited,
                missing_numbers=verdict.missing_numbers,
                dropped=self.mode == "strict",
            )
        return ClaimVerdict(
            text=verdict.text,
            markers=verdict.markers,
            coverage=verdict.coverage,
            supported=False,
            reason=verdict.reason or "reassigned",
            corrected=(best_index,),
            best_marker=best_index,
            best_coverage=best_coverage,
            inherited=verdict.inherited,
            missing_numbers=verdict.missing_numbers,
        )

    def _rebuild(
        self,
        segments: Sequence[_Segment],
        markers_by_segment: dict[int, tuple[int, ...] | None],
        dropped: set[int],
    ) -> str:
        parts: list[str] = []
        for index, segment in enumerate(segments):
            if index in dropped:
                parts.append(segment.sep)
                continue
            new_markers = markers_by_segment.get(index)
            if new_markers is None:
                parts.append(segment.raw)
                continue
            # 不 rstrip：模型原文里 [1] 前面的空格保留，只重排编号
            body = CITATION_RE.sub("", segment.body)
            if not body.strip():
                parts.append(segment.sep)
                continue
            parts.append(body + "".join(f"[{number}]" for number in new_markers) + segment.sep)
        rebuilt = "".join(parts)
        rebuilt = re.sub(r"\n{3,}", "\n\n", rebuilt)
        rebuilt = re.sub(r"[ \t]{2,}", " ", rebuilt)
        # 模型有时把 [1] 单独写一行，重排编号后就成了孤立的标记行：并回上一行末
        return tidy_marker_lines(rebuilt).strip()


def verify_citations(
    text: str,
    blocks: Sequence[ContextBlock],
    *,
    checker: ClaimChecker,
) -> ClaimCheckResult:
    """便捷函数：等价于 ``checker.verify(text, blocks)``。"""
    return checker.verify(text, blocks)
