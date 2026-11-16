"""解析器测试：MD / TXT / DOCX / XLSX / PDF 各自的章节与表格还原。"""

from __future__ import annotations

from pathlib import Path

import pytest

import gen_sample_docs
from rag_assistant.exceptions import ParseError, UnsupportedFormatError
from rag_assistant.ingestion.models import (
    ParsedDoc,
    TableBlock,
    TextBlock,
    make_doc_id,
    normalize_text,
)
from rag_assistant.ingestion.parsers import get_parser, guess_heading, supported_extensions


def _parse(path: Path, doc_type: str = "policy") -> ParsedDoc:
    parser = get_parser(path)
    return parser.parse(path, root=path.parent, doc_type=doc_type)


# --------------------------------------------------------------------------
# 标题识别（纯函数）
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("line", "expected_level"),
    [
        ("第一章 总则", 1),
        ("第二节 注册", 2),
        ("第七条 休学条件", 3),
        ("一、适用范围", 2),
        ("1.2 报销标准", 2),
        ("**第三章 休学**", 1),
    ],
)
def test_guess_heading_levels(line: str, expected_level: int) -> None:
    """各类编号都能识别出合理层级。"""
    guessed = guess_heading(line)
    assert guessed is not None
    assert guessed[0] == expected_level


@pytest.mark.parametrize(
    "line",
    [
        "学生应当在开学前两周内完成注册，逾期不予补办。",
        "这是一段很长的正文说明文字，明显不是标题。" * 3,
        "",
    ],
)
def test_guess_heading_rejects_body_text(line: str) -> None:
    """正文句子不应被误判为标题。"""
    assert guess_heading(line) is None


def test_supported_extensions_registered() -> None:
    """五种格式都已注册。"""
    exts = supported_extensions()
    for ext in ("pdf", "docx", "md", "txt", "xlsx"):
        assert ext in exts


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------
def test_markdown_structure(repo_root: Path) -> None:
    """MD：多级标题 + 表格 + 元数据。"""
    path = repo_root / "data" / "raw" / "学生学籍管理规定.md"
    doc = _parse(path)

    assert doc.doc_name == "学生学籍管理规定.md"
    assert len(doc.sha256) == 64
    assert doc.doc_type == "policy"
    assert doc.doc_id == make_doc_id(doc.rel_path)

    paths = doc.section_paths()
    assert any("第一章 总则" in p for p in paths)
    assert any("第三章 休学、复学与退学" in p and "第七条 休学条件" in p for p in paths), paths
    assert paths[0].startswith("郑州大学学生学籍管理规定")

    # 表格被识别为 TableBlock 且保留了行列
    tables = [b for s in doc.sections for b in s.blocks if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].rows[0][0] == "事项"
    assert "教务处学籍科" in tables[0].to_markdown()

    stats = doc.stats()
    assert stats["sections"] > 10
    assert stats["tables"] == 1


# --------------------------------------------------------------------------
# TXT
# --------------------------------------------------------------------------
def test_text_structure(repo_root: Path) -> None:
    """TXT：中文编号标题 + 段落累积。"""
    path = repo_root / "data" / "raw" / "办事指南-在读证明.txt"
    doc = _parse(path, doc_type="guide")

    paths = doc.section_paths()
    # 首行的文档标题被提升为一级，各"一、二、"章节嵌套其下
    assert paths[0] == "郑州大学办事指南：在读证明开具 > 一、事项名称"
    assert any("六、注意事项" in p for p in paths)
    body = "\n".join(b.text for s in doc.sections for b in s.blocks if isinstance(b, TextBlock))
    assert "工本费 2 元" in body
    assert doc.doc_type == "guide"


# --------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------
def test_docx_structure(tmp_path: Path) -> None:
    """DOCX：Heading 样式驱动的三级层级 + 表格整体保留。"""
    path = gen_sample_docs.build_docx(tmp_path / "科研经费报销管理办法.docx")
    doc = _parse(path)

    paths = doc.section_paths()
    assert any(p == "郑州大学科研经费报销管理办法" for p in paths)
    assert any("第一章 总则" in p and "第一条 适用范围" in p for p in paths), paths
    assert any("第二章 报销标准" in p for p in paths)

    tables = [b for s in doc.sections for b in s.blocks if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].rows[0] == ("费用类型", "报销限额", "审批人", "办理时限")
    assert len(tables[0].rows) == 4


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------
def test_xlsx_structure(tmp_path: Path) -> None:
    """XLSX：每个 sheet 一个章节 + 一张表。"""
    path = gen_sample_docs.build_xlsx(tmp_path / "差旅与会议费标准表.xlsx")
    doc = _parse(path)

    assert len(doc.sections) == 2
    assert doc.section_paths() == ["工作表：差旅住宿费标准", "工作表：会议费标准"]
    first = doc.sections[0].blocks[0]
    assert isinstance(first, TableBlock)
    assert first.header[0] == "地区类别"
    assert len(first.rows) == 6
    assert "省内" in first.to_markdown()


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def test_pdf_text_and_page(pdf_file: Path) -> None:
    """PDF：文本可提取、页码写入章节、不会因无中文标题而崩溃。"""
    doc = _parse(pdf_file)
    text = "\n".join(b.text for s in doc.sections for b in s.blocks if isinstance(b, TextBlock))
    assert "registrar" in text.lower()
    assert doc.sections[0].page == 1
    assert doc.sections[0].path == ("正文",)  # 英文编号非中文规则 → 走兜底章节


# --------------------------------------------------------------------------
# 异常路径
# --------------------------------------------------------------------------
def test_unsupported_extension(tmp_path: Path) -> None:
    """不支持的格式给出明确异常。"""
    path = tmp_path / "a.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        get_parser(path)


def test_empty_file_raises(tmp_path: Path) -> None:
    """空文件不应写进索引。"""
    path = tmp_path / "empty.md"
    path.write_bytes(b"")
    with pytest.raises(ParseError):
        _parse(path)


def test_normalize_text() -> None:
    """全角转半角、去零宽字符、压缩空行。"""
    raw = "第一条　学生注册\r\n\r\n\r\n应使用１号窗口\u200b办理  \n"
    cleaned = normalize_text(raw)
    assert "１号" not in cleaned
    assert "1号" in cleaned
    assert "\u200b" not in cleaned
    assert "\n\n\n" not in cleaned
