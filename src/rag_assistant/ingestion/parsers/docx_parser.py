"""DOCX 解析器。

依赖 ``python-docx``。按 body 子元素的**原始顺序**遍历段落与表格，
这样"表 1 之后的说明文字"不会被挪到表格前面。

标题识别优先级：
1. 段落样式名为 ``Heading N`` / ``标题 N`` → 直接用该层级
2. 段落的大纲级别（``w:outlineLvl``）→ 层级 = 值 + 1
3. 都没有时用 :func:`guess_heading` 的编号启发式
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..models import Section
from ...exceptions import ParseError
from .base import BaseParser, SectionBuilder, guess_heading, register_parser

_HEADING_STYLE = re.compile(r"^(?:heading|标题)\s*([1-9])$", re.IGNORECASE)


def _iter_block_items(document) -> Iterator[tuple[str, object]]:
    """按文档顺序产出 ``("paragraph", Paragraph)`` / ``("table", Table)``。"""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield "paragraph", Paragraph(child, document)
        elif tag == "tbl":
            yield "table", Table(child, document)


def _heading_level(paragraph) -> int | None:
    """从样式名或大纲级别推断标题层级，非标题返回 None。"""
    style_name = (getattr(paragraph.style, "name", "") or "").strip()
    match = _HEADING_STYLE.match(style_name)
    if match:
        return int(match.group(1))

    # outlineLvl: 0 表示一级标题
    try:
        p_pr = paragraph._p.pPr
        if p_pr is not None and p_pr.outlineLvl is not None:
            value = int(p_pr.outlineLvl.val)
            if 0 <= value <= 8:
                return value + 1
    except (AttributeError, ValueError, TypeError):
        pass
    return None


def _is_bold_short(paragraph) -> bool:
    """整段加粗且很短 —— 常见的手工"标题"写法。"""
    text = (paragraph.text or "").strip()
    if not text or len(text) > 40:
        return False
    runs = [r for r in paragraph.runs if (r.text or "").strip()]
    if not runs:
        return False
    return all(bool(r.bold) for r in runs)


@register_parser
class DocxParser(BaseParser):
    """把 DOCX 转成章节树，表格整体保留。"""

    name = "docx"
    extensions = ("docx",)

    def parse_sections(self, path: Path) -> list[Section]:
        try:
            import docx  # noqa: F401  延迟导入，未安装时只影响该格式
        except ImportError as exc:  # pragma: no cover
            raise ImportError("解析 DOCX 需要 python-docx：pip install python-docx") from exc

        from docx import Document

        try:
            document = Document(str(path))
        except Exception as exc:  # noqa: BLE001 - 损坏 zip / 非 docx 都归为解析失败
            raise ParseError(f"DOCX 打开失败（可能已损坏或不是 docx）：{path.name} -> {exc}") from exc
        builder = SectionBuilder()
        paragraph_buffer: list[str] = []

        def flush() -> None:
            if paragraph_buffer:
                builder.add_text("\n".join(paragraph_buffer))
                paragraph_buffer.clear()

        for kind, item in _iter_block_items(document):
            if kind == "table":
                flush()
                rows = [
                    [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    for row in item.rows  # type: ignore[attr-defined]
                ]
                builder.add_table(rows)
                continue

            paragraph = item
            text = (paragraph.text or "").strip()
            if not text:
                continue

            level = _heading_level(paragraph)
            if level is None and _is_bold_short(paragraph):
                guessed = guess_heading(text)
                level = guessed[0] if guessed else 4
            if level is None:
                guessed = guess_heading(text)
                # 只对"独占一行且非句子"的编号行做章节切分
                if guessed is not None and len(paragraph_buffer) == 0:
                    level = guessed[0]

            if level is not None:
                flush()
                builder.start(level, text)
            else:
                paragraph_buffer.append(text)

        flush()
        return builder.build()


__all__ = ["DocxParser"]
