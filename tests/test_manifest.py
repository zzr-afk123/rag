"""增量清单测试：扫描、差异比对、原子读写。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rag_assistant.exceptions import ManifestError
from rag_assistant.ingestion.manifest import (
    Manifest,
    apply_plan,
    diff_manifest,
    load_manifest,
    scan_directory,
    save_manifest,
)


@pytest.fixture()
def kb_dir(tmp_path: Path) -> Path:
    """构造一个包含新旧/临时/不支持格式文件的目录。"""
    root = tmp_path / "raw"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# 第一章 总则\n\n第一条 内容。", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("一、事项\n\n说明文字。", encoding="utf-8")
    (root / "~$临时.docx").write_bytes(b"lock")
    (root / "表格.csv").write_text("a,b", encoding="utf-8")
    (root / ".hidden.md").write_text("隐藏", encoding="utf-8")
    return root


def test_scan_filters_and_ids(kb_dir: Path) -> None:
    """只收白名单格式，忽略临时文件/隐藏文件，doc_id 稳定。"""
    scan = scan_directory(kb_dir)
    names = sorted(info.rel_path for info in scan.values())
    assert names == ["a.md", Path("sub/b.txt").as_posix()]

    again = scan_directory(kb_dir)
    assert {k: v.sha256 for k, v in scan.items()} == {k: v.sha256 for k, v in again.items()}


def test_scan_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="文档目录不存在"):
        scan_directory(tmp_path / "nope")


def test_diff_added_modified_removed(kb_dir: Path) -> None:
    """新增/变更/删除/未变 四种状态判定正确。"""
    scan = scan_directory(kb_dir)
    first = diff_manifest(Manifest(), scan)
    assert len(first.added) == 2
    assert not first.modified and not first.removed
    assert not first.is_empty

    # 仅改 mtime，内容不变 → 仍是未变
    manifest = apply_plan(Manifest(), first, {doc_id: 3 for doc_id in scan})
    os.utime(kb_dir / "a.md", (1, 1))
    scan2 = scan_directory(kb_dir)
    plan2 = diff_manifest(manifest, scan2)
    assert plan2.is_empty
    assert len(plan2.unchanged) == 2

    # 改内容 → 变更
    (kb_dir / "a.md").write_text("# 第一章 总则\n\n第一条 新内容。", encoding="utf-8")
    scan3 = scan_directory(kb_dir)
    plan3 = diff_manifest(manifest, scan3)
    assert len(plan3.modified) == 1
    assert plan3.modified[0].rel_path == "a.md"

    # 删文件 → 删除
    (kb_dir / "sub" / "b.txt").unlink()
    scan4 = scan_directory(kb_dir)
    plan4 = diff_manifest(manifest, scan4)
    assert len(plan4.removed) == 1


def test_apply_plan_skips_failed_docs(kb_dir: Path) -> None:
    """写入失败的文档不更新清单（下次会重试），成功的不受影响。"""
    scan = scan_directory(kb_dir)
    plan = diff_manifest(Manifest(), scan)
    successful = {list(scan)[0]: 5}
    manifest = apply_plan(Manifest(), plan, successful)

    assert manifest.doc_count == 1
    assert manifest.chunk_count == 5
    # 失败的那个仍会被下次 diff 认为是新增
    retry = diff_manifest(manifest, scan)
    assert len(retry.added) == 1


def test_save_and_load_roundtrip(tmp_path: Path, kb_dir: Path) -> None:
    """清单落盘后可原样读回，且无残留临时文件。"""
    path = tmp_path / "index" / "manifest.json"
    scan = scan_directory(kb_dir)
    manifest = apply_plan(Manifest(), diff_manifest(Manifest(), scan), {k: 2 for k in scan})
    save_manifest(path, manifest)

    assert path.exists()
    assert not list(path.parent.glob(".manifest-*"))

    loaded = load_manifest(path)
    assert loaded.doc_count == manifest.doc_count
    assert loaded.chunk_count == manifest.chunk_count
    assert loaded.updated_at  # 写入时自动打时间戳
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1


def test_load_missing_manifest_is_empty(tmp_path: Path) -> None:
    """首次运行：清单不存在时返回空清单而不是报错。"""
    manifest = load_manifest(tmp_path / "manifest.json")
    assert manifest.doc_count == 0


def test_load_corrupt_manifest(tmp_path: Path) -> None:
    """清单损坏时给出明确异常（避免静默全量重建）。"""
    path = tmp_path / "manifest.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="清单 JSON 损坏"):
        load_manifest(path)


def test_manifest_version_guard(tmp_path: Path) -> None:
    """未来版本的清单不能被旧代码误读。"""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 99, "docs": {}}), encoding="utf-8")
    with pytest.raises(ManifestError, match="高于当前支持"):
        load_manifest(path)


def test_removed_docs_dropped(kb_dir: Path) -> None:
    """删除的文档从清单里彻底移除。"""
    scan = scan_directory(kb_dir)
    plan = diff_manifest(Manifest(), scan)
    manifest = apply_plan(Manifest(), plan, {k: 1 for k in scan})
    assert manifest.doc_count == 2

    (kb_dir / "a.md").unlink()
    scan2 = scan_directory(kb_dir)
    plan2 = diff_manifest(manifest, scan2)
    updated = apply_plan(manifest, plan2, {})
    assert updated.doc_count == 1
