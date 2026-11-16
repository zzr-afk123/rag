"""表格按行分组切分（table_max_rows）的测试。

背景：整表一块时，"省内住宿费限额是多少"这类**行级提问**会被同表其他行稀释，
实测 Top-1 命中同表的另一行。按行分组、每组重复表头后，行级匹配更集中。
"""

from __future__ import annotations

from rag_assistant.config import ChunkingConfig
from rag_assistant.ingestion.chunker import Chunker
from rag_assistant.ingestion.models import ParsedDoc, Section


def make_doc(sections: list[Section]) -> ParsedDoc:
    return ParsedDoc(
        doc_id="doc-table",
        doc_name="差旅与会议费标准表.xlsx",
        source_path="/tmp/差旅与会议费标准表.xlsx",
        rel_path="差旅与会议费标准表.xlsx",
        doc_type="policy",
        sha256="b" * 64,
        size_bytes=100,
        mtime=0.0,
        sections=sections,
    )


def _section(rows: list[list[str]]) -> Section:
    section = Section(path=("工作表：差旅住宿费标准",), level=1)
    section.add_table(rows, caption="差旅住宿费标准")
    return section


def test_large_table_is_split_by_rows() -> None:
    """超过 table_max_rows 的表按组切分，每片都带表头。"""
    rows = [["地区类别", "住宿费限额"]] + [[f"地区{i}", f"{i}00 元"] for i in range(1, 6)]
    config = ChunkingConfig(max_tokens=700, min_tokens=120, overlap_tokens=60, table_max_rows=2)
    chunks = Chunker(config).chunk(make_doc([_section(rows)]))

    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(tables) == 3  # 2 + 2 + 1
    assert all("地区类别" in chunk.text for chunk in tables), "每片必须重复表头"
    assert sum("（表续" in chunk.text for chunk in tables) == 2


def test_small_table_stays_whole() -> None:
    """行数不超过阈值时保持整表，不制造碎片。"""
    rows = [["地区类别", "住宿费限额"], ["省内", "500 元"], ["省外", "600 元"]]
    config = ChunkingConfig(max_tokens=700, min_tokens=120, overlap_tokens=60, table_max_rows=4)
    chunks = Chunker(config).chunk(make_doc([_section(rows)]))

    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(tables) == 1
    assert "省内" in tables[0].text and "省外" in tables[0].text


def test_table_rows_limit_can_be_disabled() -> None:
    """table_max_rows=0 时恢复"整表不拆"的老行为。"""
    rows = [["地区类别", "住宿费限额"]] + [[f"地区{i}", f"{i}00 元"] for i in range(1, 9)]
    config = ChunkingConfig(max_tokens=700, min_tokens=120, overlap_tokens=60, table_max_rows=0)
    chunks = Chunker(config).chunk(make_doc([_section(rows)]))

    tables = [chunk for chunk in chunks if chunk.chunk_type == "table"]
    assert len(tables) == 1
    assert "地区8" in tables[0].text
