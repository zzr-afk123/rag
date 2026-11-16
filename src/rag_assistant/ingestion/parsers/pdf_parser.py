"""PDF 解析器（基于 pypdf）。

已知限制（会在 README 与报告里如实标注）：
- pypdf 只能拿到**纯文本**，拿不到字号/字体等版式信息，所以标题层级只能靠
  "第X章 / 一、/ 1.2" 这类编号启发式 + 行长判断，**准确率明显低于 DOCX**。
  因此 PDF 的切块退化为"按页 + 段落长度"，溯源以页码为主。
- 扫描件（图片型 PDF）无文本层，解析结果为空 → :meth:`ParsedDoc.validate`
  会报错并跳过该文件（不做 OCR，不在本期范围内）。
- 跨页表格无法还原为表格结构，会被当作正文段落；仅对"紧邻的连续文本块"
  做拼接。
- 重复出现在多数页面页眉/页脚的文字会被自动剔除（降低噪声对召回的干扰）。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from ...exceptions import ParseError
from ..models import Section, normalize_text
from .base import BaseParser, SectionBuilder, guess_heading, register_parser

logger = logging.getLogger("rag_assistant.ingestion.pdf")

_PAGE_NUMBER_LINE = re.compile(r"^[-—–\s]*(?:第\s*)?\d{1,3}\s*(?:页)?(?:\s*/\s*\d{1,3})?[-—–\s]*$")
_DIGITS = re.compile(r"\d+")


def _looks_like_page_number(line: str) -> bool:
    """``- 3 -`` / ``第 3 页`` / ``3 / 20`` 这类页码行。"""
    text = line.strip()
    return bool(text) and len(text) <= 12 and bool(_PAGE_NUMBER_LINE.match(text))


def _collect_repeated_lines(pages: list[str], threshold: float = 0.5) -> set[str]:
    """找出在多数页面重复出现的行（页眉/页脚），返回需剔除的规范化文本集合。"""
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    for text in pages:
        seen: set[str] = set()
        for line in text.splitlines():
            key = _DIGITS.sub("#", normalize_text(line))
            if 0 < len(key) <= 40:
                seen.add(key)
        counter.update(seen)
    limit = max(2, int(len(pages) * threshold))
    return {key for key, count in counter.items() if count >= limit}


@register_parser
class PdfParser(BaseParser):
    """把 PDF 按页解析为章节（页码写入元数据）。"""

    name = "pdf"
    extensions = ("pdf",)

    def parse_sections(self, path: Path) -> list[Section]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("解析 PDF 需要 pypdf：pip install pypdf") from exc

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                # 只尝试空密码（很多"加密"PDF 只是权限限制）
                if not reader.decrypt(""):
                    raise ParseError(f"PDF 已加密且无法用空密码打开：{path.name}")
            page_texts = [(page.extract_text() or "") for page in reader.pages]
        except ParseError:
            raise
        except Exception as exc:  # noqa: BLE001 - pypdf 的异常类型不稳定
            raise ParseError(f"PDF 解析失败：{path.name} -> {exc}") from exc

        if not any(t.strip() for t in page_texts):
            raise ParseError(f"PDF 无文本层（可能是扫描件）：{path.name}")

        repeated = _collect_repeated_lines(page_texts)
        builder = SectionBuilder()
        paragraph: list[str] = []

        def flush(page: int) -> None:
            if paragraph:
                builder.add_text("\n".join(paragraph), page=page)
                paragraph.clear()

        for page_index, text in enumerate(page_texts, start=1):
            for raw_line in text.splitlines():
                line = normalize_text(raw_line)
                if not line or _looks_like_page_number(line):
                    continue
                if _DIGITS.sub("#", line) in repeated:
                    continue

                guessed = guess_heading(line)
                if guessed is not None and len(paragraph) == 0:
                    level, title = guessed
                    builder.start(level, title, page=page_index)
                    continue
                paragraph.append(line)
            flush(page_index)
            # 每页结束视为段落边界，避免跨页整段粘在一起
            if paragraph:
                flush(page_index)

        sections = builder.build()
        if not sections:
            raise ParseError(f"PDF 解析结果为空：{path.name}")
        logger.debug("PDF 解析完成：%s，%d 页，%d 章节", path.name, len(page_texts), len(sections))
        return sections


__all__ = ["PdfParser"]
