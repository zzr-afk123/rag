"""同一份文件的多版本处理：只索引最新的一版。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.ingestion.manifest import FileInfo, make_doc_id
from rag_assistant.ingestion.versions import (
    drop_superseded,
    family_key,
    has_version_marker,
    select_latest,
)
from rag_assistant.kb import KnowledgeBase


def _info(rel: str, mtime: float = 0.0) -> FileInfo:
    return FileInfo(
        doc_id=make_doc_id(rel),
        rel_path=rel,
        abs_path=rel,
        doc_name=Path(rel).name,
        sha256=f"sha-{rel}",
        size_bytes=1,
        mtime=mtime,
    )


def test_family_key_ignores_year_and_suffix() -> None:
    assert family_key("办事指南-在读证明-2025版.txt") == family_key("办事指南-在读证明-2026版.txt")
    assert family_key("学生学籍管理规定（2025年修订）.md") == family_key("学生学籍管理规定（2026年修订）.md")
    assert family_key("实验室安全管理规定V2.docx") == family_key("实验室安全管理规定V3.docx")
    # 不相干的两份文件不应被合并
    assert family_key("图书馆管理办法.md") != family_key("学生学籍管理规定.md")


def test_marker_detection() -> None:
    assert has_version_marker("办事指南-2025版.txt")
    assert has_version_marker("规定V2.docx")
    assert has_version_marker("规定（修订版）.docx")
    assert has_version_marker("办事指南.txt", "2026/办事指南.txt")
    assert not has_version_marker("学生学籍管理规定.md")


def test_keep_newest_year() -> None:
    entries = {
        "a": _info("办事指南-在读证明-2025版.txt", mtime=200.0),
        "b": _info("办事指南-在读证明-2026版.txt", mtime=100.0),
    }
    superseded = select_latest(entries)
    assert superseded == {"a": "b"}


def test_year_in_directory() -> None:
    entries = {
        "old": _info("2025/评奖办法.md", mtime=999.0),
        "new": _info("2026/评奖办法.md", mtime=1.0),
    }
    assert select_latest(entries) == {"old": "new"}


def test_same_year_uses_mtime() -> None:
    entries = {
        "a": _info("评奖办法-2026版.txt", mtime=10.0),
        "b": _info("评奖办法-2026修订版.txt", mtime=20.0),
    }
    # 年份都是 2026，靠文件修改时间决定谁是最新
    assert select_latest(entries) == {"a": "b"}


def test_without_marker_no_grouping() -> None:
    """只有扩展名不同、没有任何版本标记的文件共存，不能被误删。"""
    entries = {
        "md": _info("学生学籍管理规定.md"),
        "docx": _info("学生学籍管理规定.docx"),
    }
    assert select_latest(entries) == {}
    kept, superseded = drop_superseded(entries)
    assert superseded == {}
    assert set(kept) == {"md", "docx"}


def test_drop_superseded_filters_entries() -> None:
    entries = {
        "old": _info("差旅费标准-2025.xlsx"),
        "new": _info("差旅费标准-2026.xlsx"),
        "other": _info("图书馆管理办法.md"),
    }
    kept, superseded = drop_superseded(entries)
    assert set(kept) == {"new", "other"}
    assert superseded == {"old": "new"}


def _two_versions(raw: Path) -> None:
    base = (raw / "办事指南-在读证明.txt").read_text(encoding="utf-8")
    (raw / "办事指南-在读证明-2025版.txt").write_text(
        base.replace("1 个工作日内审核", "3 个工作日内审核"), encoding="utf-8"
    )
    (raw / "办事指南-在读证明-2026版.txt").write_text(base, encoding="utf-8")


def test_sync_indexes_only_latest_version(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    before = kb.sync()
    baseline = kb.status()["documents"]
    assert before.ok, before.failed

    # 目录里原本就有无版本号的「办事指南-在读证明.txt」，它和 2025/2026 两版属于同一族
    _two_versions(sample_raw_dir)
    report = kb.sync()

    assert report.ok, report.failed
    assert report.superseded == 2
    assert {item["superseded_by"] for item in report.superseded_docs} == {"办事指南-在读证明-2026版.txt"}
    # 原件被移除、2025 版跳过、2026 版进来 → 文档总数不变
    assert report.removed >= 1
    assert kb.status()["documents"] == baseline

    result = kb.search("在读证明几个工作日能办完", top_n=5, threshold=0.0)
    names = {chunk.doc_name for chunk in result.chunks}
    assert "办事指南-在读证明-2026版.txt" in names
    assert "办事指南-在读证明-2025版.txt" not in names
    assert "办事指南-在读证明.txt" not in names


def test_superseded_document_removed_from_index(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """先关掉开关让三份都入库，再打开 → 被取代的两份要像删除一样清掉。"""
    _two_versions(sample_raw_dir)
    loose = replace(
        mock_settings,
        sync=replace(mock_settings.sync, prefer_latest_version=False),
    )
    kb = KnowledgeBase(loose)
    kb.sync()
    both = kb.status()["documents"]

    strict = replace(
        mock_settings,
        sync=replace(mock_settings.sync, prefer_latest_version=True),
    )
    kb2 = KnowledgeBase(strict)
    report = kb2.sync()

    assert report.removed >= 2
    assert report.superseded == 2
    assert kb2.status()["documents"] == both - 2
    result = kb2.search("在读证明几个工作日能办完", top_n=5, threshold=0.0)
    assert all(chunk.doc_name != "办事指南-在读证明-2025版.txt" for chunk in result.chunks)
    assert all(chunk.doc_name != "办事指南-在读证明.txt" for chunk in result.chunks)
