"""增量清单（manifest）。

职责：记录"上一次成功入库时每个文件长什么样"，用于下次扫描时判断
新增 / 变更 / 删除 / 未变。清单本身是 ``data/index/manifest.json``：

    {
      "version": 1,
      "updated_at": "2026-10-28T10:00:00+08:00",
      "docs": {
        "<doc_id>": {
          "rel_path": "学生学籍管理规定.md",
          "doc_name": "学生学籍管理规定.md",
          "sha256": "...",
          "file_version": "...",
          "chunks": 12,
          "indexed_at": "..."
        }
      }
    }

为什么用 doc_id（路径哈希）做键而不是文件名：改名即"删旧+增新"，
行为确定，不会出现"两个文件抢同一个 doc_id"的脏数据。
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..exceptions import ManifestError
from .models import make_doc_id, make_file_version

MANIFEST_VERSION = 1
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime(_TIME_FORMAT)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 sha256（大文件不整读进内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FileInfo:
    """扫描得到的一个源文件快照。"""

    doc_id: str
    rel_path: str
    abs_path: str
    doc_name: str
    sha256: str
    size_bytes: int
    mtime: float

    @property
    def file_version(self) -> str:
        return make_file_version(self.sha256, self.mtime)

    def to_entry(self, chunk_count: int) -> "ManifestEntry":
        return ManifestEntry(
            rel_path=self.rel_path,
            doc_name=self.doc_name,
            sha256=self.sha256,
            file_version=self.file_version,
            size_bytes=self.size_bytes,
            mtime=self.mtime,
            chunks=chunk_count,
            indexed_at=_now(),
        )


def scan_directory(
    root: Path,
    *,
    include_patterns: Sequence[str] = ("*.pdf", "*.docx", "*.md", "*.txt", "*.xlsx"),
    exclude_dirs: Sequence[str] = ("__pycache__", ".git", "~$"),
) -> dict[str, FileInfo]:
    """扫描目录，返回 ``{doc_id: FileInfo}``。

    跳过：不在 include_patterns 内的文件、隐藏文件（``~$`` 开头是 Office 临时文件）、
    exclude_dirs 指定的目录。
    """
    root = Path(root)
    if not root.exists():
        raise ManifestError(f"文档目录不存在：{root}")
    if not root.is_dir():
        raise ManifestError(f"文档目录不是文件夹：{root}")

    result: dict[str, FileInfo] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in exclude_dirs or part.startswith("~$") or part.startswith(".") for part in rel.split("/")[:-1]):
            continue
        if path.name.startswith(("~$", ".")):
            continue
        if not any(fnmatch.fnmatch(path.name.lower(), pattern.lower()) for pattern in include_patterns):
            continue

        info = FileInfo(
            doc_id=make_doc_id(rel),
            rel_path=rel,
            abs_path=str(path.resolve()),
            doc_name=path.name,
            sha256=sha256_of_file(path),
            size_bytes=path.stat().st_size,
            mtime=path.stat().st_mtime,
        )
        result[info.doc_id] = info
    return result


# --------------------------------------------------------------------------
# 清单读写
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ManifestEntry:
    """清单中的单文档记录。"""

    rel_path: str
    doc_name: str
    sha256: str
    file_version: str
    size_bytes: int = 0
    mtime: float = 0.0
    chunks: int = 0
    indexed_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "rel_path": self.rel_path,
            "doc_name": self.doc_name,
            "sha256": self.sha256,
            "file_version": self.file_version,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "chunks": self.chunks,
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ManifestEntry":
        return cls(
            rel_path=str(data.get("rel_path", "")),
            doc_name=str(data.get("doc_name", "")),
            sha256=str(data.get("sha256", "")),
            file_version=str(data.get("file_version", "")),
            size_bytes=int(data.get("size_bytes", 0) or 0),
            mtime=float(data.get("mtime", 0.0) or 0.0),
            chunks=int(data.get("chunks", 0) or 0),
            indexed_at=str(data.get("indexed_at", "")),
        )


@dataclass
class Manifest:
    """整份清单。"""

    version: int = MANIFEST_VERSION
    updated_at: str = ""
    docs: dict[str, ManifestEntry] = field(default_factory=dict)

    @property
    def doc_count(self) -> int:
        return len(self.docs)

    @property
    def chunk_count(self) -> int:
        return sum(entry.chunks for entry in self.docs.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "updated_at": self.updated_at or _now(),
            "docs": {doc_id: entry.to_dict() for doc_id, entry in self.docs.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "Manifest":
        version = int(data.get("version", MANIFEST_VERSION) or MANIFEST_VERSION)
        if version > MANIFEST_VERSION:
            raise ManifestError(f"清单版本 {version} 高于当前支持的 {MANIFEST_VERSION}")
        raw_docs = data.get("docs") or {}
        if not isinstance(raw_docs, Mapping):
            raise ManifestError("清单 docs 字段格式非法")
        docs = {
            str(doc_id): ManifestEntry.from_dict(entry)
            for doc_id, entry in raw_docs.items()
            if isinstance(entry, Mapping)
        }
        return cls(version=version, updated_at=str(data.get("updated_at", "")), docs=docs)


def load_manifest(path: Path) -> Manifest:
    """读取清单；文件不存在时返回空清单（首次运行属于正常情况）。"""
    path = Path(path)
    if not path.exists():
        return Manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"清单 JSON 损坏：{path} -> {exc}") from exc
    if not isinstance(data, Mapping):
        raise ManifestError(f"清单顶层必须是对象：{path}")
    return Manifest.from_dict(data)


def save_manifest(path: Path, manifest: Manifest) -> None:
    """原子写入清单（先写临时文件再 replace，避免中途崩溃留下半截 JSON）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = _now()
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:  # noqa: BLE001 - 失败时清理临时文件后原样抛出
        Path(tmp_name).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# 差异比对
# --------------------------------------------------------------------------
@dataclass
class SyncPlan:
    """一次同步要做的动作。"""

    added: list[FileInfo] = field(default_factory=list)
    modified: list[FileInfo] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # doc_id 列表
    unchanged: list[str] = field(default_factory=list)  # doc_id 列表

    @property
    def touched(self) -> list[FileInfo]:
        return [*self.added, *self.modified]

    @property
    def is_empty(self) -> bool:
        return not self.added and not self.modified and not self.removed

    def summary(self) -> str:
        return (
            f"新增 {len(self.added)} / 变更 {len(self.modified)} / "
            f"删除 {len(self.removed)} / 未变 {len(self.unchanged)}"
        )


def diff_manifest(manifest: Manifest, current: Mapping[str, FileInfo]) -> SyncPlan:
    """对比清单与最新扫描结果，产出同步计划。

    判定规则：
    - doc_id 不在清单 → 新增
    - doc_id 在清单但 sha256 不同 → 变更（内容变了才重建，改 mtime 不算）
    - doc_id 在清单但扫描不到 → 删除
    - 否则未变
    """
    plan = SyncPlan()
    for doc_id, info in current.items():
        entry = manifest.docs.get(doc_id)
        if entry is None:
            plan.added.append(info)
        elif entry.sha256 != info.sha256:
            plan.modified.append(info)
        else:
            plan.unchanged.append(doc_id)
    plan.removed = [doc_id for doc_id in manifest.docs if doc_id not in current]
    plan.added.sort(key=lambda i: i.rel_path)
    plan.modified.sort(key=lambda i: i.rel_path)
    plan.removed.sort()
    plan.unchanged.sort()
    return plan


def apply_plan(
    manifest: Manifest,
    plan: SyncPlan,
    chunk_counts: Mapping[str, int],
) -> Manifest:
    """根据已成功入库的结果更新清单。

    ``chunk_counts``: ``{doc_id: 入库 chunk 数}``，只应包含真正成功写库的文档。
    失败的文件不应出现在这里，清单里也就保持旧状态（下次继续重试）。
    """
    docs = dict(manifest.docs)
    for info in [*plan.added, *plan.modified]:
        count = chunk_counts.get(info.doc_id)
        if count is None:
            continue
        docs[info.doc_id] = info.to_entry(count)
    for doc_id in plan.removed:
        docs.pop(doc_id, None)
    return Manifest(version=MANIFEST_VERSION, updated_at=_now(), docs=docs)


def prune_missing(manifest: Manifest, keep_ids: Iterable[str]) -> Manifest:
    """只保留指定 doc_id（用于"向量库重建后对齐清单"的修复场景）。"""
    keep = set(keep_ids)
    return Manifest(
        version=MANIFEST_VERSION,
        updated_at=_now(),
        docs={doc_id: entry for doc_id, entry in manifest.docs.items() if doc_id in keep},
    )
