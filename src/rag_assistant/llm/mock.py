"""确定性 Mock LLM（无网络、无 Key）。

为什么需要它：需求要求"无 Key 时可用 mock 模式跑通基础流程与测试"。
但 mock 必须能**分别模拟三种任务**（改写 / 回答 / 摘要），否则链路上层
（rewriter / answerer / session）就没法被真实驱动。

做法：system 消息以 ``[[TASK:xxx]]`` 开头（见 ``prompts.py``），这里按标记分派。
输出是规则生成的，**它不是语义模型**：
- 改写：用"上一问 + 当前追问 + 疑问词替换"模拟指代消解，只够验证链路；
- 回答：按问题词面相关度挑整句原文，分条列出并附真实编号，保证引用校验能通过；
- 摘要：截断拼接，不做真正的信息压缩。
所有"看起来合理"的地方都是刻意设计的可断言行为，不是模型能力。
"""

from __future__ import annotations

import re
from typing import Iterator, Sequence

from ..tokens import count_tokens
from .base import LLMClient, LLMResponse, LLMMessage, LLMUsage
from .prompts import TASK_ANSWER, TASK_REWRITE, TASK_SUMMARY, task_of

#: 指代开头
_PRONOUN = re.compile(r"^(那么|那|这个|那个|该|上述|此|它|其中|这些|那些)[的]?")

#: 从上下文块里抽取编号、文档、章节
_BLOCK_HEADER = re.compile(r"^\[(\d{1,2})\]\s*文档：(.+?)\s*｜\s*章节：(.+?)(?:\s*｜\s*页码：(.+))?$")
_QUESTION_LINE = re.compile(r"^【问题】\s*\n(.+)$", re.S)
_HISTORY_USER = re.compile(r"用户：(.+?)(?:\n|$)")

#: 句子切分：保留标点，中文标点为主，英文标点兜底；条目编号（一、/（二）/1.）前也切开
_SENTENCE_CUT = re.compile(
    r"(?<=[。；！？!?;])|(?=[一二三四五六七八九十]{1,2}、)|(?=（[一二三四五六七八九十]{1,2}）)"
)

#: 句首的条目编号，如"一、""（二）""3."
_ENUM_LEAD = re.compile(r"^(?:[一二三四五六七八九十]{1,2}、|（[一二三四五六七八九十]{1,2}）|\d{1,2}[.、])\s*")

#: 段落里单独成句的"标题式"前缀（如 PDF 正文开头的文档名）
_TITLE_TAIL = re.compile(r"[^\s。；！？]{0,40}$")


def _bigrams(text: str) -> set[str]:
    cleaned = "".join(text.split()).lower()
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _overlap(query: str, text: str) -> float:
    """字符二元组覆盖率（mock 用来"挑最相关的资料"）。"""
    grams = _bigrams(query)
    if not grams:
        return 0.0
    return len(grams & _bigrams(text)) / len(grams)


def _sentences(text: str) -> list[str]:
    # PDF 抽取的换行是排版换行，不是段落分隔：中文直接拼回去，避免句中混入空格
    body = re.sub(r"[ \t]*\n[ \t]*", "", text).strip()
    body = re.sub(r"[ \t]{2,}", " ", body)
    if not body:
        return []
    return [piece.strip() for piece in _SENTENCE_CUT.split(body) if piece.strip()]


def _trim_leading_title(sentence: str, doc: str) -> str:
    """资料正文常常以文档标题开头，标题单独成句时要剥掉。

    否则会出现"根据《大学生反诈告知书.pdf》正文，大学生反诈宣传告知书 当前……"这种复读。
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", doc).strip()
    for candidate in (doc.strip(), stem):
        if candidate and sentence.startswith(candidate):
            rest = sentence[len(candidate) :].lstrip(" 　·—：:》】]）)")
            if rest:
                return rest
    # PDF 抽取常把标题和正文挤在一行：标题后紧跟空格，且标题里没有标点
    head, sep, rest = sentence.partition("　" if "　" in sentence else " ")
    if sep and len(head) <= 20 and len(rest) >= 24 and not re.search(r"[，。；：、！？!?;,]", head):
        return rest
    return sentence


def _clip(sentence: str, max_chars: int) -> str:
    """超长句子按标点截断，避免在词中间硬切。"""
    if len(sentence) <= max_chars:
        return sentence
    head = sentence[:max_chars]
    for sep in ("，", "；", "、", ","):
        index = head.rfind(sep)
        if index >= max_chars // 2:
            return head[: index + 1]
    return head.rstrip("，；、, ") + "…"


def _strip_title_line(body: str) -> str:
    """PDF 抽取的第一行常常只有文档标题，单独成行；它不是可引用内容。"""
    first, sep, rest = body.partition("\n")
    if sep and len(first) <= 30 and not re.search(r"[。；！？，,：:]", first) and rest.strip():
        return rest.strip()
    return body


def _pick_sentences(
    question: str,
    body: str,
    doc: str,
    *,
    limit: int = 3,
    max_chars: int = 96,
) -> list[str]:
    """挑与问题词面最相关的整句原文，按原文顺序返回。"""
    sentences = [_trim_leading_title(item, doc) for item in _sentences(body)]
    cleaned: list[str] = []
    for item in sentences:
        item = _ENUM_LEAD.sub("", item).strip()
        if not item:
            continue
        # "现特作如下告知："这类只有引导、没有内容的分句直接丢掉
        if item.endswith(("：", ":", "，", ",")) and len(item) < 40:
            continue
        # 分句末尾残留的引导语（"……现特作如下告知："）也切掉
        trimmed = re.sub(r"(?:现|特|具体作出|具体|如下)[^，,；;。]{0,10}[：:]$", "", item).strip()
        if trimmed:
            item = trimmed
            # 引导语连同前面的句号一起被切掉时，补回一个句号
            if item[-1] not in "。；！？!?;…":
                item += "。"
        # 纯标题式短句（"校园常见诈骗类型""在读证明（中英文）开具"）当要点没有意义：
        # 字数少、没有逗号分句、也不带数字，基本就是个小标题
        if len(item) <= 14 and not re.search(r"[，,；;：:\d]", item):
            continue
        cleaned.append(_clip(item, max_chars))
    if not cleaned:
        return []
    if not question:
        return cleaned[:limit]

    ranked = sorted(
        range(len(cleaned)),
        key=lambda index: (-_overlap(question, cleaned[index]), index),
    )
    picked = sorted(ranked[:limit])
    return [cleaned[index] for index in picked]


def _table_snippet(text: str, max_chars: int = 140) -> str:
    """表格取"表头 + 首行数据"并逐列配对，避免只给表头等于没给内容。"""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    rows = [
        line
        for line in lines
        if line.startswith("|") and not re.fullmatch(r"[|\-: ]+", line)
    ]
    if len(rows) < 2:
        return " ".join(lines)[:max_chars]

    header = [cell.strip() for cell in rows[0].strip("|").split("|")]
    data = [cell.strip() for cell in rows[1].strip("|").split("|")]
    pairs = [f"{name} {value}".strip() for name, value in zip(header, data) if name or value]
    snippet = "；".join(pairs) if pairs else rows[1]
    return snippet[:max_chars]


class MockLLM(LLMClient):
    """规则驱动的假 LLM。"""

    model = "mock-llm"

    def __init__(self, refuse_text: str = "当前知识库中未找到相关依据，建议咨询对应部门或人工客服。") -> None:
        self._refuse_text = refuse_text
        self.calls: list[str] = []  # 测试用：记录每次调用的任务类型

    # ---- 接口 ----
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        task = task_of(messages)
        self.calls.append(task)
        text = self._respond(task, messages)
        prompt_tokens = sum(count_tokens(message.content) for message in messages)
        return LLMResponse(
            text=text,
            usage=LLMUsage(prompt_tokens=prompt_tokens, completion_tokens=count_tokens(text)),
            model=self.model,
            finish_reason="stop",
        )

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        text = self.complete(messages, temperature=temperature, max_tokens=max_tokens).text
        step = 18
        for start in range(0, len(text), step):
            yield text[start : start + step]

    @property
    def is_mock(self) -> bool:
        return True

    # ---- 任务分派 ----
    def _respond(self, task: str, messages: Sequence[LLMMessage]) -> str:
        if task == TASK_REWRITE:
            return self._rewrite(messages)
        if task == TASK_ANSWER:
            return self._answer(messages)
        if task == TASK_SUMMARY:
            return self._summary(messages)
        return ""

    def _rewrite(self, messages: Sequence[LLMMessage]) -> str:
        """模拟指代消解。

        规则（刻意保守）：只有"很短的问题"才拼接上一轮问题；
        长问题本身通常已经自足（"那复学呢？需要什么材料？"），拼接反而会
        把两个主题混在一起，稀释检索。这一点在真模型上同样需要注意。
        """
        payload = messages[-1].content
        question_match = re.search(r"当前问题：(.+?)(?:\n|$)", payload)
        question = question_match.group(1).strip() if question_match else ""
        if not question:
            return '{"standalone": "", "variants": []}'

        history = _HISTORY_USER.findall(payload)
        stripped = _PRONOUN.sub("", question).strip() or question
        standalone = stripped
        # 只有"指代开头 + 去掉指代后极短"才算真的追问（如"那它的期限呢？"）。
        # 像"那复学呢？需要什么材料？"自身已带主语，拼接反而混入上一轮话题。
        if history and _PRONOUN.match(question) and len(stripped) <= 8:
            previous = history[-1].strip().rstrip("？?。 ")
            standalone = f"{previous} {question}".strip()

        # mock 不产出等价查询：规则造不出有意义的改写，硬造只会污染 RRF 融合。
        # multi-query 链路由 tests/test_rewriter.py 的桩 LLM 覆盖。
        import json

        return json.dumps({"standalone": standalone, "variants": []}, ensure_ascii=False)

    def _answer(self, messages: Sequence[LLMMessage]) -> str:
        """挑最相关的两块资料，按原文整句列要点并附真实编号。

        输出结构（刻意做成"可读的答案"而不是原始段落）::

            根据《学生学籍管理规定.md》第三章 休学、复学与退学 > 第七条 休学条件：
            · 学生因病需停课治疗的应当办理休学。[1]
            · 休学期一般以一年为期，累计不得超过两年。[1]

        引用编号必须落在每句末尾，闸门 2 与 claim 校验都依赖它。
        """
        payload = messages[-1].content
        question_match = _QUESTION_LINE.search(payload)
        question = question_match.group(1).strip() if question_match else ""

        blocks: list[dict[str, str]] = []
        current: dict[str, str] | None = None
        for line in payload.splitlines():
            stripped = line.strip()
            header = _BLOCK_HEADER.match(stripped)
            if header:
                number, doc, section, page = header.groups()
                current = {
                    "number": number,
                    "doc": doc.strip(),
                    "section": section.strip(),
                    "page": (page or "无").strip(),
                    "body": "",
                }
                blocks.append(current)
                continue
            # 段落结束标记：问题区、最近对话区
            if stripped.startswith("【问题】") or stripped.startswith("【最近对话】"):
                current = None
                continue
            if current is None:
                continue
            if stripped.startswith("正文："):
                current["body"] = stripped[len("正文：") :]
            elif current.get("body") or stripped.startswith("|"):
                # 正文可能被换行拆成多行（PDF 一行一段），要全部接上，否则只剩标题
                current["body"] = (current["body"] + "\n" + stripped).strip()

        # 只有标题/一句定语的短段（"一、事项名称：在读证明（中英文）开具。"）单独成段时，
        # 把它当答案只会答非所问，直接排除
        usable = [block for block in blocks if len(block.get("body", "").strip()) >= 24]
        if not usable:
            return self._refuse_text

        # 只选与问题有字面重合的块，最多两块，并按原始（检索）顺序输出。
        # 完全无关的块要丢掉——把它塞进答案正是"问反诈答出实验室条款"的根因之一。
        # 同分时优先正文更长的块：短段往往只是标题，信息量不足以作答。
        scored = [(block, _overlap(question, block["body"]), len(block["body"])) for block in usable]
        relevant = [item for item in scored if item[1] > 0]
        best = sorted(relevant or scored[:1], key=lambda item: (item[1], item[2]), reverse=True)[:2]
        # 第二段只有相关度和第一段接近时才带进来：差得远的多半是碰巧撞上两三个字的
        # 无关条款（"复学材料"带出"缓交学费"表就是这种）
        top = best[0][1] if best else 0
        best = [item for item in best if item[1] >= top * 0.75]
        chosen = {id(block) for block, _, _ in best}
        selected = [block for block in usable if id(block) in chosen]

        parts: list[str] = []
        seen: set[str] = set()
        for block in selected:
            body = _strip_title_line(block["body"])
            number = block["number"]
            section = block["section"]
            heading = f"《{block['doc']}》" if not section or section == "正文" else f"《{block['doc']}》{section}"
            lead = "根据" if not parts else "另据"

            if body.lstrip().startswith("|"):
                snippet = _table_snippet(body)
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    parts.append(f"{lead}{heading}：{snippet}[{number}]")
                continue

            picked = _pick_sentences(question, body, block["doc"])
            picked = [item for item in picked if item not in seen]
            if not picked:
                continue
            seen.update(picked)
            parts.append(f"{lead}{heading}：")
            for sentence in picked:
                parts.append(f"· {sentence}[{number}]")

        if not parts:
            return self._refuse_text
        return "\n".join(parts)

    def _summary(self, messages: Sequence[LLMMessage]) -> str:
        payload = messages[-1].content
        body = payload.split("待压缩的对话：", 1)[-1]
        compact = " ".join(body.split())
        return compact[:150] if compact else "（无有效对话）"
