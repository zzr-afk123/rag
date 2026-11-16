"""文档版本识别：同一份文件放了多个年份/版本时，只索引最新的一份。

适用场景：``办事指南-在读证明-2025版.txt`` 与 ``办事指南-在读证明-2026版.txt``
同时放进 ``data/raw``。两版内容差异往往只在个别数字（时限、金额、电话），
检索时分数几乎一样，模型会随机引用其中一版——问"多久能办完"可能答出旧规定。

判定规则（全部集中在这里，便于调整）：

1. **文档族**：去掉文件名里的年份、``V2`` 这类版本号、``修订版/终稿/最新版``
   这类修饰词与分隔符后，剩下的字符串相同即视为同一份文件的不同版本；
   年份也允许出现在**目录名**里（``2025/办事指南.pdf`` 与 ``2026/办事指南.pdf``）。
2. **只有出现明确的版本标记才合并**：``学生学籍管理规定.md`` 与
   ``学生学籍管理规定.docx`` 这种只是扩展名不同、没有任何版本标记的，不合并。
3. **最新**：年份大的优先；年份相同（或都没写年份）时按文件修改时间，
   再相同则按相对路径，保证结果确定、不随扫描顺序变化。

被取代的文件不会入库；如果它已经在索引里，同步时会像"文件被删除"一样清掉
它的向量与 BM25 条目，并在 :class:`SyncReport` 里记一笔。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping

from .manifest import FileInfo

# 四位数年份（19xx / 20xx），前后不能紧挨着数字，避免把 12025 这类编号当成年份
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# V1 / v2.1 / V3 这类版本号
_VER = re.compile(r"(?<![A-Za-z0-9])[vV]\s*(\d+(?:\.\d+)*)(?![0-9])")
# 常见的版本修饰词：出现即说明"这是同一份文件的某一版"
_NOISE = re.compile(
    r"(修订版|修订稿|修订|终稿|最终版|最新版|最新|试行版|试行|征求意见稿|定稿|旧版|新版|发布版|正式版|备份|副本|copy)",
    re.IGNORECASE,
)
# 分隔符与括号
_SEP = re.compile(r"[\s\-_—–·.()（）【】\[\]]+")
# 去掉分隔符后残留的"版/稿"后缀（"…2026版" 与 "…2026修订版" 应归为同一族）
_SUFFIX = re.compile(r"[版稿]+$")


@dataclass(frozen=True)
class VersionInfo:
    """一个文件参与版本比较所需的元数据。"""

    doc_id: str
    rel_path: str
    family: str
    year: int
    marker: bool

    @property
    def rank(self) -> tuple[int, float, str]:
        return (self.year, 0.0, self.rel_path)


def _stem(name: str) -> str:
    return PurePosixPath(name).stem


def _years(*texts: str) -> int:
    found = [int(match) for text in texts for match in _YEAR.findall(text)]
    return max(found) if found else 0


def family_key(name: str) -> str:
    """把文件名收敛成"文档族"键：去掉年份、版本号、版本修饰词与分隔符。"""
    text = _stem(name)
    text = _YEAR.sub("", text)
    text = _VER.sub("", text)
    text = _NOISE.sub("", text)
    text = _SEP.sub("", text)
    text = _SUFFIX.sub("", text)
    return text.strip().lower()


def has_version_marker(name: str, rel_path: str = "") -> bool:
    """文件名或目录名里是否出现了年份 / V2 / 修订版这类版本标记。"""
    text = f"{_stem(name)} {rel_path}"
    return bool(_YEAR.search(text) or _VER.search(text) or _NOISE.search(text))


def version_info(info: FileInfo) -> VersionInfo:
    """从 :class:`FileInfo` 提取版本元数据（年份可来自目录名）。"""
    rel = PurePosixPath(info.rel_path)
    dirs = " ".join(rel.parts[:-1])
    year = max(_years(_stem(info.doc_name)), _years(dirs))
    return VersionInfo(
        doc_id=info.doc_id,
        rel_path=info.rel_path,
        family=family_key(info.doc_name),
        year=year,
        marker=has_version_marker(info.doc_name, info.rel_path),
    )


def select_latest(entries: Mapping[str, FileInfo]) -> dict[str, str]:
    """返回 ``{被取代的 doc_id: 保留的 doc_id}``。

    只对"同一文档族 ≥2 份、且其中至少一份带版本标记"的族做取舍，
    其余文件原样保留，因此不会误伤"同名不同格式"这类正常共存的文件。
    """
    groups: dict[str, list[tuple[str, FileInfo, VersionInfo]]] = {}
    for doc_id, info in entries.items():
        meta = version_info(info)
        if not meta.family:
            continue
        groups.setdefault(meta.family, []).append((doc_id, info, meta))

    superseded: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2 or not any(meta.marker for _, _, meta in members):
            continue
        # 年份大的优先；年份相同按 mtime；再相同按路径，保证结果确定
        ranked = sorted(
            members,
            key=lambda item: (item[2].year, item[1].mtime, item[1].rel_path),
            reverse=True,
        )
        keeper_id = ranked[0][0]
        for doc_id, info, _meta in ranked[1:]:
            # 同族里内容完全相同的两份（比如 2025/ 与 2026/ 目录下的同一份文件）
            # 也按"保留最新"处理，避免同一份资料在上下文里出现两次
            superseded[doc_id] = keeper_id
    return superseded


def drop_superseded(entries: Mapping[str, FileInfo]) -> tuple[dict[str, FileInfo], dict[str, str]]:
    """过滤掉被更新版本取代的文件，返回 ``(保留的, {被取代 doc_id: 保留 doc_id})``。"""
    superseded = select_latest(entries)
    if not superseded:
        return dict(entries), {}
    kept = {doc_id: info for doc_id, info in entries.items() if doc_id not in superseded}
    return kept, superseded
