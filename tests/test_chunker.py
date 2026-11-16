"""切块测试：章节切分、overlap、表格整体保留、元数据完整性。"""

from __future__ import annotations

import pytest

from rag_assistant.config import ChunkingConfig
from rag_assistant.ingestion.chunker import Chunker, _text_units
from rag_assistant.ingestion.models import ParsedDoc, Section
from rag_assistant.tokens import count_tokens, tail_tokens

BASE_SENTENCE = "学生应当在开学前两周内完成学期注册，逾期未注册的按自动退学处理。"


def repeat_to_tokens(target: int) -> str:
    """生成约 target token 的中文文本（尽可能接近目标值）。"""
    pieces: list[str] = []
    total = 0
    while total < target:
        pieces.append(BASE_SENTENCE)
        total += count_tokens(BASE_SENTENCE)
    return "".join(pieces)


def make_doc(sections: list[Section], *, name: str = "测试制度.md") -> ParsedDoc:
    """构造一个最小 ParsedDoc（不经过解析器）。"""
    return ParsedDoc(
        doc_id="docid000000000001",
        doc_name=name,
        source_path="/tmp/" + name,
        rel_path=name,
        doc_type="policy",
        sha256="a" * 64,
        size_bytes=1234,
        mtime=1_700_000_000.0,
        sections=sections,
        parser="test",
    )


@pytest.fixture()
def chunk_config() -> ChunkingConfig:
    return ChunkingConfig(max_tokens=200, min_tokens=40, overlap_tokens=30, keep_tables_whole=True, table_max_tokens=400)


# --------------------------------------------------------------------------
# 长度与 overlap
# --------------------------------------------------------------------------
def test_chunk_respects_max_tokens(chunk_config: ChunkingConfig) -> None:
    """长章节被切成多块，每块不超过 max_tokens，且都带章节路径。"""
    section = Section(path=("第三章 学籍管理", "第七条 注册"), level=2)
    for _ in range(4):
        section.add_text(repeat_to_tokens(100))
    doc = make_doc([section])

    chunks = Chunker(chunk_config).chunk(doc)

    assert len(chunks) >= 4
    for chunk in chunks:
        assert chunk.tokens <= chunk_config.max_tokens
        assert count_tokens(chunk.text) == chunk.tokens
        assert chunk.section_path == "第三章 学籍管理 > 第七条 注册"
        assert chunk.text.strip()


def test_overlap_between_neighbours(chunk_config: ChunkingConfig) -> None:
    """相邻 chunk 之间保留 overlap（对齐整句）。"""
    section = Section(path=("第二章",), level=1)
    for _ in range(4):
        section.add_text(repeat_to_tokens(120))
    doc = make_doc([section])

    chunks = Chunker(chunk_config).chunk(doc)
    assert len(chunks) >= 2

    expected = tail_tokens(chunks[0].text, chunk_config.overlap_tokens)
    assert expected, "overlap 文本不应为空"
    assert chunks[1].text.startswith(expected)


def test_no_overlap_when_configured_zero() -> None:
    """overlap=0 时相邻块严格不重复。"""
    config = ChunkingConfig(max_tokens=120, min_tokens=0, overlap_tokens=0)
    section = Section(path=("第一章",), level=1)
    for _ in range(3):
        section.add_text(repeat_to_tokens(90))
    chunks = Chunker(config).chunk(make_doc([section]))

    assert len(chunks) >= 2
    assert not chunks[1].text.startswith(chunks[0].text[-40:])


def test_short_tail_merged() -> None:
    """尾部过短的碎片回并到前一块，避免"一句话 chunk"。"""
    base_tokens = count_tokens(BASE_SENTENCE)
    config = ChunkingConfig(max_tokens=base_tokens * 5 + 10, min_tokens=40, overlap_tokens=0)
    section = Section(path=("第一条",), level=1)
    section.add_text(BASE_SENTENCE * 5)  # 刚好塞满一块
    section.add_text("本条自发布之日起施行。")  # 溢出且过短
    chunks = Chunker(config).chunk(make_doc([section]))

    assert len(chunks) == 1
    assert "自发布之日起施行" in chunks[0].text


# --------------------------------------------------------------------------
# 表格
# --------------------------------------------------------------------------
def test_table_kept_whole(chunk_config: ChunkingConfig) -> None:
    """小表格整块保留，不跟正文混在一个 chunk 里。"""
    section = Section(path=("第三章",), level=1)
    section.add_text("材料清单如下：")
    section.add_table(
        [["事项", "部门"], ["注册", "教务处"], ["缴费", "财务处"]],
        caption="表1 办理事项",
    )
    section.add_text("以上事项均可线上办理。")
    chunks = Chunker(chunk_config).chunk(make_doc([section]))

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    text = table_chunks[0].text
    assert "事项" in text and "教务处" in text and "财务处" in text
    assert "表1 办理事项" in text
    # 表格块不应包含正文段落
    assert "以上事项均可线上办理" not in text


def test_large_table_split_with_repeated_header() -> None:
    """超大表格按行拆分，但每片都重复表头。"""
    config = ChunkingConfig(max_tokens=200, min_tokens=40, overlap_tokens=0, table_max_tokens=300)
    rows = [["地区类别", "报销限额", "备注"]]
    for index in range(60):
        rows.append([f"地区{index}", f"{index * 100}元", repeat_to_tokens(10)])
    section = Section(path=("报销标准",), level=1)
    section.add_table(rows, caption="差旅费标准")
    chunks = Chunker(config).chunk(make_doc([section]))

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        assert chunk.text.count("地区类别") == 1  # 每片都有表头且只出现一次
        assert "|" in chunk.text


def test_text_units_split_oversized_paragraph() -> None:
    """单段超长时按句子切，且不产生超过 max_tokens 的单元。"""
    paragraph = repeat_to_tokens(500)
    units = _text_units(paragraph, max_tokens=150)
    assert len(units) > 1
    assert all(count_tokens(u) <= 150 for u in units)
    # 拼回去内容不丢
    assert "".join(units).replace("\n", "") == paragraph


# --------------------------------------------------------------------------
# 元数据与确定性
# --------------------------------------------------------------------------
def test_metadata_complete(chunk_config: ChunkingConfig) -> None:
    """每个 chunk 都带完整溯源元数据，可直接写向量库。"""
    section = Section(path=("第四章 毕业",), level=1, page=None)
    section.add_text(repeat_to_tokens(80))
    chunks = Chunker(chunk_config).chunk(make_doc([section]))
    meta = chunks[0].to_metadata()

    for key in (
        "doc_id",
        "doc_name",
        "doc_type",
        "section_path",
        "order",
        "chunk_type",
        "page",
        "source_path",
        "file_version",
        "tokens",
    ):
        assert key in meta
    assert meta["page"] == 0  # 无页码统一写 0（Chroma 不接受 None）
    assert chunks[0].page_label == "无页码"
    # embed_text 前置文档名与章节，便于提升检索主题一致性
    assert meta["doc_name"] in chunks[0].embed_text
    assert "第四章 毕业" in chunks[0].embed_text


def test_chunking_is_deterministic(chunk_config: ChunkingConfig) -> None:
    """同一份文档两次切块结果完全一致（chunk_id 可作为幂等键）。"""
    section = Section(path=("第一条",), level=1)
    for _ in range(3):
        section.add_text(repeat_to_tokens(90))
    doc = make_doc([section])
    first = Chunker(chunk_config).chunk(doc)
    second = Chunker(chunk_config).chunk(doc)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith(doc.doc_id) for cid in ids)


def test_chunk_ids_change_when_text_changes(chunk_config: ChunkingConfig) -> None:
    """正文改动 → chunk_id 变化（日志里能定位"哪块变了"）。"""
    section_a = Section(path=("第一条",), level=1)
    section_a.add_text("本条自 2024 年 9 月 1 日起施行。")
    section_b = Section(path=("第一条",), level=1)
    section_b.add_text("本条自 2025 年 9 月 1 日起施行。")

    chunker = Chunker(chunk_config)
    first = chunker.chunk(make_doc([section_a]))[0]
    second = chunker.chunk(make_doc([section_b]))[0]
    assert first.chunk_id != second.chunk_id


def test_empty_section_skipped(chunk_config: ChunkingConfig) -> None:
    """没有内容的章节不产生 chunk。"""
    empty = Section(path=("空章节",), level=1)
    filled = Section(path=("有内容",), level=1)
    filled.add_text("本章有效。")
    chunks = Chunker(chunk_config).chunk(make_doc([empty, filled]))
    assert len(chunks) == 1
    assert chunks[0].section_path == "有内容"
