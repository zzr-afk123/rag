"""Markdown 解析器：ATX 标题（# ~ ######）+ 管道表格 + 代码块。

额外兼容"**第三章 学籍管理**"这种整行加粗的伪标题（很多学院文档是这么写的）。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import Section
from .base import BaseParser, SectionBuilder, guess_heading, register_parser

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_TABLE_SEPARATOR = re.compile(r"^[\s|:，,\-—]+$")
_BOLD_ONLY = re.compile(r"^\*\*(.+?)\*\*$")


def _split_table_row(line: str) -> list[str]:
    """切分一行管道表格（忽略行首尾的竖线）。"""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [cell.strip() for cell in text.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    """判断是否是 Markdown 表格的分隔行（|---|---|）。"""
    if not cells:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip()) and any(c.strip() for c in cells)


@register_parser
class MarkdownParser(BaseParser):
    """把 Markdown 转成章节树。"""

    name = "markdown"
    extensions = ("md", "markdown")

    def parse_sections(self, path: Path) -> list[Section]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        builder = SectionBuilder()
        paragraph: list[str] = []
        in_code = False
        i = 0

        def flush() -> None:
            if paragraph:
                builder.add_text("\n".join(paragraph))
                paragraph.clear()

        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()

            # ---- 代码块：原样保留为正文，不参与表格/标题识别 ----
            if stripped.startswith("```"):
                in_code = not in_code
                paragraph.append(stripped)
                i += 1
                continue
            if in_code:
                paragraph.append(raw)
                i += 1
                continue

            if not stripped:
                flush()
                i += 1
                continue

            # ---- ATX 标题 ----
            heading = _ATX_HEADING.match(stripped)
            if heading:
                flush()
                level = len(heading.group(1))
                builder.start(level, heading.group(2))
                i += 1
                continue

            # ---- 整行加粗的伪标题 ----
            bold = _BOLD_ONLY.match(stripped)
            if bold and len(paragraph) == 0:
                guessed = guess_heading(stripped)
                if guessed is not None:
                    level, title = guessed
                    builder.start(level, title)
                    i += 1
                    continue

            # ---- 管道表格 ----
            if "|" in stripped and i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if "|" in next_stripped and _TABLE_SEPARATOR.match(next_stripped) and "-" in next_stripped:
                    flush()
                    table_rows: list[list[str]] = [_split_table_row(stripped)]
                    i += 2  # 跳过表头与分隔行
                    while i < len(lines):
                        row_line = lines[i].strip()
                        if not row_line or "|" not in row_line:
                            break
                        cells = _split_table_row(row_line)
                        if not _is_separator_row(cells):
                            table_rows.append(cells)
                        i += 1
                    builder.add_table(table_rows)
                    continue

            paragraph.append(raw.rstrip())
            i += 1

        flush()
        return builder.build()


__all__ = ["MarkdownParser"]
