"""结构化切块。

策略（对应需求 1.切块策略）：
1. **优先按章节切**：一个 :class:`Section` 的正文尽量落在同一个 chunk 里，
   天然带 ``第三章 > 第五条`` 这种章节路径，引用溯源最准。
2. **超长再切**：章节正文超过 ``max_tokens`` 时按句边界贪心打包，
   相邻 chunk 之间保留 ``overlap_tokens`` 的重叠（对齐到整句，不切半句）。
3. **表格不拆散**：整表成一个 chunk；只有当单表超过 ``table_max_tokens``
   时才按行分组，且每组都重复表头，保证任何一片单独看都能读懂。
4. 章节尾部过短的碎片会尝试回并到前一块，避免出现"只有一句话"的低质量 chunk。

chunk_id 生成规则 ``{doc_id}#{序号:04d}-{正文hash8}``：序号保证顺序稳定，
hash 让同一位置内容变化时 id 也随之变化（便于日志里定位"哪块变了"）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ..config import ChunkingConfig
from ..exceptions import ChunkingError
from ..tokens import count_tokens, split_sentences, tail_tokens
from .models import Chunk, ChunkType, ParsedDoc, Section, TableBlock, TextBlock

#: 回并碎片时允许的松弛系数（合并后最多超出 max_tokens 的 20%）
_MERGE_SLACK = 1.2


@dataclass(frozen=True)
class ChunkingStats:
    """切块统计，用于 dry-run 输出与日志。"""

    docs: int = 0
    sections: int = 0
    chunks: int = 0
    table_chunks: int = 0
    tokens: int = 0

    def merge(self, other: "ChunkingStats") -> "ChunkingStats":
        return ChunkingStats(
            docs=self.docs + other.docs,
            sections=self.sections + other.sections,
            chunks=self.chunks + other.chunks,
            table_chunks=self.table_chunks + other.table_chunks,
            tokens=self.tokens + other.tokens,
        )


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """把超长"单句"按比例硬切（极少发生，仅用于异常长的无标点段落）。"""
    total = count_tokens(text)
    if total <= max_tokens:
        return [text]
    pieces: list[str] = []
    start = 0
    length = len(text)
    step = max(1, int(length * max_tokens / total))
    while start < length:
        piece = text[start : start + step]
        if piece.strip():
            pieces.append(piece.strip())
        start += step
    return pieces


def _text_units(text: str, max_tokens: int) -> list[str]:
    """把一段正文拆成"不超过 max_tokens 的打包单元"。

    先按换行分自然段；单段超长再按句子贪心打包；单句仍超长才硬切。
    保持这个粒度是为了让上层能按段落/句子边界拼接，不产生半句。
    """
    units: list[str] = []
    for paragraph in text.split("\n"):
        para = paragraph.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            units.append(para)
            continue

        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in split_sentences(para):
            size = count_tokens(sentence)
            if size > max_tokens:
                if buffer:
                    units.append("".join(buffer))
                    buffer, buffer_tokens = [], 0
                units.extend(_hard_split(sentence, max_tokens))
                continue
            if buffer and buffer_tokens + size > max_tokens:
                units.append("".join(buffer))
                buffer, buffer_tokens = [sentence], size
            else:
                buffer.append(sentence)
                buffer_tokens += size
        if buffer:
            units.append("".join(buffer))
    return units


def _table_pieces(block: TableBlock, config: ChunkingConfig) -> list[tuple[str, ChunkType]]:
    """表格切分：默认整表保留；超长或超行数才按行分组并重复表头。

    为什么还要按**行数**切（``chunking.table_max_rows``）：整表一块时，
    "省内住宿费限额是多少"这种行级提问会被同表其他行稀释，
    实测 Top-1 命中同表的另一行。按 4 行一组切开后，每块自带表头，
    行级问题的向量/词面匹配都更集中。
    """
    markdown = block.to_markdown()
    if not markdown:
        return []

    header = list(block.header)
    body = [list(row) for row in block.rows[1:]]
    too_many_rows = config.table_max_rows > 0 and len(body) > config.table_max_rows
    too_long = count_tokens(markdown) > config.table_max_tokens
    if not header or not body or not (too_long or too_many_rows) or not config.keep_tables_whole:
        return [(markdown, "table")]

    caption = block.caption or ""
    pieces: list[tuple[str, ChunkType]] = []
    group: list[list[str]] = []
    group_tokens = 0

    def render(rows: list[list[str]], suffix: str) -> str:
        from .models import render_markdown_table

        rows_with_header = [header] + rows
        text = render_markdown_table(rows_with_header, caption or None)
        return f"{text}\n{suffix}" if suffix else text

    for row in body:
        probe = render(group + [row], "")
        size = count_tokens(probe)
        full_by_rows = config.table_max_rows > 0 and len(group) >= config.table_max_rows
        full_by_tokens = bool(group) and group_tokens + size > config.table_max_tokens
        if group and (full_by_rows or full_by_tokens):
            pieces.append((render(group, f"（表续：{caption}）" if caption else "（表续）"), "table"))
            group, group_tokens = [row], size
        else:
            group.append(row)
            group_tokens = size
    if group:
        pieces.append((render(group, ""), "table"))
    return pieces


def _chunk_section(section: Section, config: ChunkingConfig) -> list[tuple[str, ChunkType]]:
    """把一个章节切成 ``(正文, 类型)`` 序列。"""
    pieces: list[tuple[str, ChunkType]] = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush(carry_overlap: bool) -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            pieces.append((text, "text"))
        if carry_overlap and config.overlap_tokens > 0:
            tail = tail_tokens(text, config.overlap_tokens)
            buffer = [tail] if tail else []
            buffer_tokens = count_tokens(tail) if tail else 0
        else:
            buffer, buffer_tokens = [], 0

    for block in section.blocks:
        if isinstance(block, TextBlock):
            for unit in _text_units(block.text, config.max_tokens):
                size = count_tokens(unit)
                if buffer and buffer_tokens + size > config.max_tokens:
                    flush(carry_overlap=True)
                    # 重叠后剩余空间仍不够，放弃重叠以保证不超长
                    if buffer and buffer_tokens + size > config.max_tokens:
                        buffer, buffer_tokens = [], 0
                buffer.append(unit)
                buffer_tokens += size
        elif isinstance(block, TableBlock):
            flush(carry_overlap=False)
            pieces.extend(_table_pieces(block, config))

    flush(carry_overlap=False)

    # 尾部碎片回并：最后一块过短时合并到前一块
    if (
        len(pieces) >= 2
        and pieces[-1][1] == "text"
        and pieces[-2][1] == "text"
        and count_tokens(pieces[-1][0]) < config.min_tokens
    ):
        tail_text = pieces[-1][0]
        prev_text = pieces[-2][0]
        joined = f"{prev_text}\n{tail_text}"
        if count_tokens(joined) <= config.max_tokens * _MERGE_SLACK:
            pieces[-2] = (joined, "text")
            pieces.pop()

    return pieces


def _looks_like_doc_title(element: str, doc_name: str) -> bool:
    """判断章节路径的首元素是不是"文档标题"（避免引用里标题出现两次）。

    很多文档的第一级标题就是文档名（``# 明德学院学生学籍管理规定``），
    而文件名是 ``学生学籍管理规定.md``——两者常常一个是另一个的超集。
    这里只做保守判断：归一化后相等，或互相包含。
    """
    stem = doc_name.rsplit(".", 1)[0]
    a = "".join(element.split())
    b = "".join(stem.split())
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _clean_section_path(section: Section, doc_name: str) -> str:
    """生成用于引用的章节路径：剥掉与文档名重复的首层标题。"""
    parts = list(section.path)
    if parts and _looks_like_doc_title(parts[0], doc_name):
        parts = parts[1:]
    return " > ".join(part for part in parts if part)


class Chunker:
    """把 :class:`ParsedDoc` 切成入库单元。"""

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        """切块入口。返回按文档内顺序排列的 chunk 列表。"""
        if not doc.sections:
            raise ChunkingError(f"{doc.doc_name} 没有可切块的章节")
        chunks: list[Chunk] = []
        order = 0
        for section in doc.sections:
            section_page = section.page or 0
            section_path = _clean_section_path(section, doc.doc_name)
            for text, chunk_type in _chunk_section(section, self._config):
                body = text.strip()
                if not body:
                    continue
                digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}#{order:04d}-{digest}",
                        doc_id=doc.doc_id,
                        doc_name=doc.doc_name,
                        doc_type=doc.doc_type,
                        section_path=section_path,
                        section_level=section.level,
                        order=order,
                        chunk_type=chunk_type,
                        text=body,
                        tokens=count_tokens(body),
                        page=section_page,
                        source_path=doc.source_path,
                        file_version=doc.file_version,
                    )
                )
                order += 1
        return chunks

    def chunk_many(self, docs: Iterable[ParsedDoc]) -> tuple[list[Chunk], ChunkingStats]:
        """批量切块，同时返回统计信息。"""
        all_chunks: list[Chunk] = []
        stats = ChunkingStats()
        for doc in docs:
            doc_chunks = self.chunk(doc)
            all_chunks.extend(doc_chunks)
            stats = stats.merge(
                ChunkingStats(
                    docs=1,
                    sections=len(doc.sections),
                    chunks=len(doc_chunks),
                    table_chunks=sum(1 for c in doc_chunks if c.chunk_type == "table"),
                    tokens=sum(c.tokens for c in doc_chunks),
                )
            )
        return all_chunks, stats
