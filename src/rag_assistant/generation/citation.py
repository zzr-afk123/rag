"""引用溯源：编号校验 + 聚合 + 重编号。

模型输出的是"资料编号"（对应 prompt 里的 [1][2]…）。这里做三件事：

1. **校验**：编号必须落在资料范围内，越界视为无效引用（这是拒答闸门 2 的判据）。
2. **聚合**（方案假设 #15）：同一份文档的**同一章节**只保留一个编号，
   因为用户关心的是"这条规定在哪一节"，不需要知道它被切成两个 chunk。
3. **重编号**：按"首次被引用"的先后顺序重新编 1..N，与最终 citations 列表严格对应。

重编号带来的额外好处：模型如果引用了重复章节，输出里的 `[1][1]` 会被折叠成 `[1]`，
阅读体验更干净。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .context import ContextBlock

#: 匹配 [1] / [12]（不匹配 markdown 链接与 [a-z]）
CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_COLLAPSE_RE = re.compile(r"(\[\d{1,2}\])(?:\s*\1)+")


@dataclass(frozen=True)
class Citation:
    """引用来源（对外返回的结构）。"""

    index: int
    doc_id: str
    doc_name: str
    section_path: str
    page: int
    chunk_id: str
    quote: str
    file_version: str = ""
    doc_type: str = ""

    @property
    def page_label(self) -> str:
        return f"第 {self.page} 页" if self.page else "无页码"

    def render_line(self) -> str:
        """一行可读来源（用于答案末尾的"引用来源"列表）。"""
        location = self.section_path or "正文"
        return f"[{self.index}] 《{self.doc_name}》{location}（{self.page_label}）：{self.quote}"

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "section_path": self.section_path,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "file_version": self.file_version,
            "doc_type": self.doc_type,
        }


@dataclass(frozen=True)
class RemapResult:
    """重编号结果。"""

    text: str
    citations: list[Citation]
    invalid_indices: list[int]

    @property
    def has_citation(self) -> bool:
        return bool(self.citations)

    @property
    def has_invalid(self) -> bool:
        return bool(self.invalid_indices)


def _snippet(text: str, limit: int) -> str:
    """取原文摘录：优先在句末标点处断开，避免半句。"""
    body = " ".join(text.split())
    if len(body) <= limit:
        return body
    window = body[:limit]
    for punctuation in ("。", "；", "！", "？", ";", "."):
        position = window.rfind(punctuation)
        if position >= limit // 2:
            return window[: position + 1]
    return window + "…"


def extract_markers(text: str) -> list[int]:
    """按出现顺序提取引用编号（含重复）。"""
    return [int(match.group(1)) for match in CITATION_RE.finditer(text or "")]


def remap_citations(
    text: str,
    blocks: Sequence[ContextBlock],
    *,
    quote_chars: int = 120,
) -> RemapResult:
    """校验并按"文档+章节"聚合、重编号。

    Args:
        text: 模型原始输出。
        blocks: 送进 prompt 的资料块（编号即 prompt 里的编号）。
        quote_chars: 引用摘录长度上限。

    Returns:
        :class:`RemapResult`；``invalid_indices`` 非空说明模型引用了不存在的资料。
    """
    by_index = {block.index: block for block in blocks}
    group_to_index: dict[tuple[str, str], int] = {}
    citations: list[Citation] = []
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        block = by_index.get(number)
        if block is None:
            if number not in invalid:
                invalid.append(number)
            return ""  # 越界编号直接抹掉，不留脏标记
        key = (block.chunk.doc_id, block.chunk.section_path)
        if key not in group_to_index:
            group_to_index[key] = len(citations) + 1
            citations.append(
                Citation(
                    index=group_to_index[key],
                    doc_id=block.chunk.doc_id,
                    doc_name=block.chunk.doc_name,
                    section_path=block.chunk.section_path,
                    page=block.chunk.page,
                    chunk_id=block.chunk.chunk_id,
                    # 摘录用**合并后的章节正文**（context.build_context 已按章节合并），
            # 否则用户看到的摘录与模型实际依据的文本不一致
            quote=_snippet(block.body or block.chunk.text, quote_chars),
                    file_version=block.chunk.file_version,
                    doc_type=block.chunk.doc_type,
                )
            )
        return f"[{group_to_index[key]}]"

    remapped = CITATION_RE.sub(replace, text or "")
    remapped = _COLLAPSE_RE.sub(r"\1", remapped)
    # 抹掉编号后可能留下多余空格/空括号
    remapped = re.sub(r"[（(]\s*[）)]", "", remapped)
    remapped = re.sub(r"[ \t]{2,}", " ", remapped).strip()
    return RemapResult(text=remapped, citations=citations, invalid_indices=invalid)


def render_citations(citations: Sequence[Citation]) -> str:
    """渲染答案末尾的来源列表（Markdown 列表）。"""
    if not citations:
        return ""
    lines = [f"- {citation.render_line()}" for citation in citations]
    # 纯文本标题（不再是 Markdown 加粗）：早期版本输出 `**引用来源**`，在纯文本
    # 终端与部分前端里会原样显示星号；这里直接给"引用来源"四个字，
    # 样式交给前端（.sources 小标题）处理。
    return "引用来源\n" + "\n".join(lines)
