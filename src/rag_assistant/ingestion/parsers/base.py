"""解析器基类与共享工具：章节栈、标题识别、解析器注册表。

五个格式（PDF/DOCX/MD/TXT/XLSX）差异很大，但"把内容归到某个章节"这件事
是共通的，因此统一用 :class:`SectionBuilder` 维护标题栈：

    标题(level=1) → 标题(level=2) → 正文
    遇到 level<=当前 的新标题时，栈弹出到合适层级再压入。

这样输出的 ``Section.path`` 就是 ``("第三章 学籍管理", "第五条 入学与注册")``
这种可直接用于引用溯源的结构化路径。
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from ...exceptions import ParseError, UnsupportedFormatError
from ..models import ParsedDoc, Section, TableBlock, make_doc_id, normalize_text

# --------------------------------------------------------------------------
# 标题识别
# --------------------------------------------------------------------------
_CN_CHAPTER = re.compile(r"^第[一二三四五六七八九十百千零〇两\d]+[章编部篇]")
_CN_SECTION = re.compile(r"^第[一二三四五六七八九十百千零〇两\d]+节")
_CN_ARTICLE = re.compile(r"^第[一二三四五六七八九十百千零〇两\d]+条")
_CN_ENUM = re.compile(r"^[一二三四五六七八九十]{1,3}\s*[、.．]\s*")
_CN_PAREN = re.compile(r"^[（(]\s*[一二三四五六七八九十\d]{1,3}\s*[）)]\s*")
_NUM_DOT = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[、.．\s]\s*")
_HEADING_MAX_CHARS = 60


def guess_heading(line: str) -> tuple[int, str] | None:
    """尝试把一行识别为标题，返回 ``(层级, 标题文本)``；不是标题返回 None。

    规则（宁可漏判，不轻易误判，误判会把正文切碎）：
    - 长度 > 60 字符、或以句号/分号结尾 → 不是标题
    - ``第X章`` → 1 级；``第X节`` → 2 级；``第X条`` → 3 级
    - ``一、`` → 2 级；``（一）`` → 4 级（且要求很短、无句末标点）
    - ``1.2.3`` → 层级 = 点段数；``1.`` → 2 级
    """
    text = line.strip()
    if not text or len(text) > _HEADING_MAX_CHARS:
        return None
    if text[-1] in "。；;：:，,":
        return None
    # Markdown 粗体独占一行常被当成小标题（"**第三章 学籍管理**"）
    bold = re.fullmatch(r"\*\*(.+?)\*\*", text)
    if bold:
        inner = bold.group(1).strip()
        if 0 < len(inner) <= _HEADING_MAX_CHARS:
            guessed = guess_heading(inner)
            return guessed if guessed else (3, inner)

    if _CN_CHAPTER.match(text):
        return 1, text
    if _CN_SECTION.match(text):
        return 2, text
    if _CN_ARTICLE.match(text):
        return 3, text
    if _CN_ENUM.match(text):
        return 2, text
    if _NUM_DOT.match(text):
        number = _NUM_DOT.match(text).group(1)  # type: ignore[union-attr]
        level = min(1 + number.count("."), 4)
        return level, text
    if _CN_PAREN.match(text) and len(text) <= 20:
        return 4, text
    return None


class SectionBuilder:
    """按标题栈累积章节，输出扁平的 :class:`Section` 列表。"""

    def __init__(self, default_title: str = "正文") -> None:
        self._default_title = default_title
        self._stack: list[tuple[int, str]] = []
        self._sections: list[Section] = []
        self._current: Section | None = None
        self._page: int | None = None

    # ---- 内部 ----
    def _ensure_section(self) -> Section:
        if self._current is None:
            self._stack = [(0, self._default_title)]
            self._current = Section(path=(self._default_title,), level=0, page=self._page)
            self._sections.append(self._current)
        return self._current

    # ---- 对外 ----
    def start(self, level: int, title: str, *, page: int | None = None) -> Section:
        """开启新章节：弹出层级 >= level 的旧标题，再压入新标题。"""
        clean = normalize_text(title)
        if not clean:
            raise ParseError("标题不能为空")
        if not self._stack:
            level = 1  # 文档首个标题一律视为顶层，避免"一、"被误判成二级
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, clean))
        self._current = Section(
            path=tuple(t for _, t in self._stack),
            level=level,
            page=page if page is not None else self._page,
        )
        self._sections.append(self._current)
        return self._current

    def set_page(self, page: int | None) -> None:
        """记录当前位置对应的页码（PDF 用）。"""
        self._page = page
        if self._current is not None and self._current.page is None:
            self._current.page = page

    def add_text(self, text: str, *, page: int | None = None) -> None:
        """向当前章节追加正文；尚无章节时自动建"正文"章节。"""
        if page is not None:
            self.set_page(page)
        self._ensure_section().add_text(text)

    def add_table(self, rows: Iterable[Iterable[str]], caption: str | None = None) -> None:
        """向当前章节追加整张表格。"""
        self._ensure_section().add_table(rows, caption)

    def build(self) -> list[Section]:
        """返回非空章节（空标题节被丢弃，其内容本来就在子节里）。"""
        result: list[Section] = []
        for section in self._sections:
            if not section.blocks:
                continue
            if len(section.path) == 1 and section.path[0] == self._default_title and not section.blocks:
                continue
            result.append(section)
        return result

    @property
    def current(self) -> Section | None:
        return self._current


# --------------------------------------------------------------------------
# 解析器基类
# --------------------------------------------------------------------------
class BaseParser(ABC):
    """所有解析器的基类。

    子类只需实现 :meth:`parse_sections`，文件哈希/元信息/自检由基类统一处理。
    """

    name: str = "base"
    extensions: Sequence[str] = ()

    @abstractmethod
    def parse_sections(self, path: Path) -> list[Section]:
        """解析文件为章节列表。不处理 doc_id/哈希等元信息。"""

    def parse(self, path: Path, *, root: Path, doc_type: str = "policy") -> ParsedDoc:
        """完整解析流程，返回带元数据的 :class:`ParsedDoc`。"""
        path = Path(path)
        if not path.is_file():
            raise ParseError(f"文件不存在：{path}")
        raw = path.read_bytes()
        if not raw:
            raise ParseError(f"文件为空：{path}")
        sha256 = hashlib.sha256(raw).hexdigest()
        try:
            rel_path = path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:  # 文件不在 root 下时退化为文件名
            rel_path = path.name

        sections = self.parse_sections(path)
        doc = ParsedDoc(
            doc_id=make_doc_id(rel_path),
            doc_name=path.name,
            source_path=str(path.resolve()),
            rel_path=rel_path,
            doc_type=doc_type,
            sha256=sha256,
            size_bytes=len(raw),
            mtime=path.stat().st_mtime,
            sections=sections,
            parser=self.name,
        )
        doc.validate()
        return doc


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------
_REGISTRY: dict[str, BaseParser] = {}


def register_parser(parser_cls: type[BaseParser]) -> type[BaseParser]:
    """把解析器**类**注册到扩展名映射（内部持有无状态实例）。

    之所以接收类而不是实例：子类用 ``@register_parser`` 装饰即可完成注册，
    调用方也不需要关心实例化。
    """
    instance = parser_cls()
    for ext in instance.extensions:
        _REGISTRY[ext.lower().lstrip(".")] = instance
    return parser_cls


def get_parser(path: str | Path) -> BaseParser:
    """按扩展名取解析器，不支持时抛 :class:`UnsupportedFormatError`。"""
    suffix = Path(path).suffix.lower().lstrip(".")
    parser = _REGISTRY.get(suffix)
    if parser is None:
        raise UnsupportedFormatError(f"不支持的文档格式：.{suffix}（支持 {sorted(_REGISTRY)}）")
    return parser


def supported_extensions() -> list[str]:
    """当前已注册的扩展名列表。"""
    return sorted(_REGISTRY)


def table_caption_from_rows(rows: Sequence[Sequence[str]]) -> str | None:
    """表格块辅助：若无 caption，用表头拼一个可读描述。"""
    if not rows:
        return None
    header = [str(c).strip() for c in rows[0] if str(c).strip()]
    return "表格：" + " / ".join(header) if header else None


def merge_table_continuation(blocks: list) -> list:
    """合并相邻的表格块（PDF 跨页表格场景），避免同一张表被切成两片。

    只合并"紧邻"的表格块；这里不做表头去重，交给上层展示时处理。
    """
    merged: list = []
    for block in blocks:
        if (
            merged
            and isinstance(block, TableBlock)
            and isinstance(merged[-1], TableBlock)
        ):
            prev: TableBlock = merged[-1]
            merged[-1] = TableBlock(rows=prev.rows + block.rows, caption=prev.caption or block.caption)
        else:
            merged.append(block)
    return merged
