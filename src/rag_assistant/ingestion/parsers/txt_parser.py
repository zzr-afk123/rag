"""纯文本解析器。

TXT 没有标题标记，只能用启发式识别（``第一章`` / ``一、`` / ``1.2`` / ``（三）``）。
识别策略保守：只看"独占一行且较短、不以句末标点结尾"的行，避免把正文里
引用的条款编号误当成新章节。
"""

from __future__ import annotations

from pathlib import Path

from ..models import Section
from .base import BaseParser, SectionBuilder, guess_heading, register_parser


@register_parser
class TxtParser(BaseParser):
    """按空行分段 + 编号启发式识别标题。"""

    name = "text"
    extensions = ("txt", "text")

    def parse_sections(self, path: Path) -> list[Section]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        builder = SectionBuilder()
        paragraph: list[str] = []

        # 首个标题之前的短行通常是文档标题（"明德学院办事指南：在读证明"），
        # 提升为一级标题，能让引用路径带上文档主题，而不是落到"正文"里。
        first_heading = next((i for i, line in enumerate(lines) if guess_heading(line.strip())), None)
        if first_heading is not None:
            leading = [line.strip() for line in lines[:first_heading] if line.strip()]
            if 0 < len(leading) <= 2 and sum(len(item) for item in leading) <= 60:
                builder.start(1, " ".join(leading))
                lines = lines[first_heading:]

        def flush() -> None:
            if paragraph:
                builder.add_text("\n".join(paragraph))
                paragraph.clear()

        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                flush()
                continue

            guessed = guess_heading(stripped)
            if guessed is not None:
                flush()
                level, title = guessed
                builder.start(level, title)
                continue

            # 全角缩进视为新段落起点（中文公文常见）
            if raw[:1] in {"　", " "} and paragraph and len(paragraph[-1]) < 30:
                flush()
            paragraph.append(stripped)

        flush()
        return builder.build()


__all__ = ["TxtParser"]
