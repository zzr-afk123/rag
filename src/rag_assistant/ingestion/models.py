"""摄入层数据模型：解析结果 → 章节 → 块 → chunk。

设计原则：
- **解析层不关心切块，切块层不关心文件格式**，两者用 :class:`ParsedDoc` 交接。
- 章节内保持原始顺序（段落与表格交错），这样切块时不会把"表下说明文字"
  挪到表格前面。
- 所有元数据字段都是标量（str/int/float），可直接写入 Chroma 的 metadata，
  不需要再做序列化。页码未知时统一写 0（Chroma 不接受 None）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Union

from ..exceptions import ParseError

BlockType = Literal["text", "table"]
ChunkType = Literal["text", "table"]

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")

# 只把全角数字/字母转半角（U+FF10-FF19 / FF21-FF3A / FF41-FF5A）。
# 不用整体 NFKC：那会把中文标点"：" "，" "（）"全变成 ASCII，
# 引用摘录里看起来会很怪，也会影响中文句号断句。
_FULLWIDTH_ALNUM = {
    **{code: code - 0xFEE0 for code in range(0xFF10, 0xFF1A)},
    **{code: code - 0xFEE0 for code in range(0xFF21, 0xFF3B)},
    **{code: code - 0xFEE0 for code in range(0xFF41, 0xFF5B)},
}


def normalize_text(raw: str) -> str:
    """统一空白与不可见字符，保留中文标点语义。

    - 全角数字/字母转半角（"１"→"1"），便于 BM25 与数字匹配；
      中文标点保持不变。
    - 去零宽字符、制表符与全角空格转普通空格、行尾空格清理、
      连续空行压缩为两个换行。
    """
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").translate(_FULLWIDTH_ALNUM)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\t", " ").replace("\u3000", " ")
    text = _TRAILING_SPACES.sub("\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def render_markdown_table(rows: Iterable[Iterable[str]], caption: str | None = None) -> str:
    """把二维单元格渲染成 Markdown 表格（首行当表头）。

    表格整体保留可让 LLM 保持"行—列"对应关系，比拍平成文本更可靠。
    空表返回空串；单行表会补一行分隔线以保持合法 Markdown。
    """
    matrix = [[normalize_text(str(cell)) if cell is not None else "" for cell in row] for row in rows]
    matrix = [row for row in matrix if any(cell for cell in row)]
    if not matrix:
        return ""
    width = max(len(row) for row in matrix)
    padded = [row + [""] * (width - len(row)) for row in matrix]
    header, *body = padded
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    table = "\n".join(lines)
    return f"{caption}\n{table}" if caption else table


def make_doc_id(rel_path: str | Path) -> str:
    """doc_id = 相对路径的 sha1 前 16 位（重命名视为删旧增新，方案假设 #11）。"""
    key = str(rel_path).replace("\\", "/").strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def make_file_version(sha256: str, mtime: float) -> str:
    """文件版本标签 = sha256 前 12 位 + 修改时间戳（秒）。"""
    return f"{sha256[:12]}-{int(mtime)}"


# --------------------------------------------------------------------------
# 解析结果
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TextBlock:
    """一个自然段/列表项/标题下的正文片段。"""

    text: str
    kind: BlockType = "text"


@dataclass(frozen=True)
class TableBlock:
    """整张表格。``rows`` 保留二维结构，切块时才可能按行拆分。"""

    rows: tuple[tuple[str, ...], ...]
    caption: str | None = None
    kind: BlockType = "table"

    def to_markdown(self) -> str:
        return render_markdown_table(self.rows, self.caption)

    @property
    def header(self) -> tuple[str, ...]:
        return self.rows[0] if self.rows else ()


Block = Union[TextBlock, TableBlock]


@dataclass
class Section:
    """一个"章节"单元：标题路径 + 该标题下的块序列。"""

    path: tuple[str, ...]
    level: int
    blocks: list[Block] = field(default_factory=list)
    page: int | None = None

    @property
    def title(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def path_str(self) -> str:
        """形如 ``第三章 学籍管理 > 第五条 入学与注册``。"""
        return " > ".join(p for p in self.path if p)

    def add_text(self, text: str) -> None:
        """追加正文块（空文本自动忽略）。"""
        cleaned = normalize_text(text)
        if cleaned:
            self.blocks.append(TextBlock(text=cleaned))

    def add_table(self, rows: Iterable[Iterable[str]], caption: str | None = None) -> None:
        """追加表格块（可迭代行）。"""
        matrix = tuple(tuple(str(c or "") for c in row) for row in rows)
        if any(any(cell.strip() for cell in row) for row in matrix):
            self.blocks.append(TableBlock(rows=matrix, caption=caption))

    @property
    def token_hint(self) -> int:
        """粗略字符量，仅用于日志统计。"""
        return sum(len(b.text) if isinstance(b, TextBlock) else len(b.to_markdown()) for b in self.blocks)


@dataclass
class ParsedDoc:
    """一份解析完成的文档。"""

    doc_id: str
    doc_name: str
    source_path: str
    rel_path: str
    doc_type: str
    sha256: str
    size_bytes: int
    mtime: float
    sections: list[Section] = field(default_factory=list)
    parser: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def file_version(self) -> str:
        return make_file_version(self.sha256, self.mtime)

    def section_paths(self) -> list[str]:
        return [s.path_str for s in self.sections]

    def validate(self) -> None:
        """解析结果自检：没有内容视为失败（避免把空文档写进索引）。"""
        if not self.sections:
            raise ParseError(f"{self.doc_name} 未解析出任何章节")
        if not any(s.blocks for s in self.sections):
            raise ParseError(f"{self.doc_name} 解析结果为空（可能是扫描件/加密文件）")

    def stats(self) -> dict[str, int]:
        text_blocks = sum(1 for s in self.sections for b in s.blocks if isinstance(b, TextBlock))
        tables = sum(1 for s in self.sections for b in s.blocks if isinstance(b, TableBlock))
        return {
            "sections": len(self.sections),
            "text_blocks": text_blocks,
            "tables": tables,
            "chars": sum(s.token_hint for s in self.sections),
        }


# --------------------------------------------------------------------------
# 切块结果
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    """一个可入库的最小检索单元，自带完整溯源元数据。"""

    chunk_id: str
    doc_id: str
    doc_name: str
    doc_type: str
    section_path: str
    section_level: int
    order: int
    chunk_type: ChunkType
    text: str
    tokens: int
    page: int
    source_path: str
    file_version: str

    def to_metadata(self) -> dict[str, str | int | float]:
        """转成向量库 metadata（全部标量，page 未知为 0）。"""
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type,
            "section_path": self.section_path,
            "section_level": self.section_level,
            "order": self.order,
            "chunk_type": self.chunk_type,
            "page": self.page,
            "source_path": self.source_path,
            "file_version": self.file_version,
            "tokens": self.tokens,
        }

    @property
    def embed_text(self) -> str:
        """送进 embedding 的文本：前置"文档名 > 章节路径"提升主题一致性。

        ``text`` 本身保持纯净（只含正文），这样引用摘录不会被前缀污染。
        """
        prefix = f"{self.doc_name} > {self.section_path}" if self.section_path else self.doc_name
        return f"{prefix}\n{self.text}"

    @property
    def page_label(self) -> str:
        return f"第 {self.page} 页" if self.page else "无页码"


def chunk_record(chunk: Chunk) -> dict[str, str | int]:
    """导出为可 JSON 序列化的记录（dry-run 输出与清单使用）。"""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "section_path": chunk.section_path,
        "chunk_type": chunk.chunk_type,
        "tokens": chunk.tokens,
        "page": chunk.page,
        "preview": chunk.text[:80].replace("\n", " "),
    }
