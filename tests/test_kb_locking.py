"""知识库同步的并发与回滚测试（阶段 E 核心）。

覆盖：
- 同步确实在写锁内提交（锁被真正使用，不是摆设）；
- 查询遇到写锁会按 read_timeout 报 409，而不是读到半成品或无限挂起；
- 严重错误时清单回滚到同步前状态；
- 分批提交（缩短写锁窗口）；
- 锁文件在同步结束后被清理。
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rag_assistant.config import Settings
from rag_assistant.exceptions import IndexBusyError
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.kb import KnowledgeBase


@pytest.fixture()
def kb(mock_settings: Settings, sample_raw_dir: Path) -> KnowledgeBase:
    return KnowledgeBase(mock_settings)


def test_second_sync_is_rejected_while_one_holds_the_task_lock(kb: KnowledgeBase) -> None:
    """CLI 与 API 是两个进程，各自的 SyncJobManager 管不到对方。

    没有整次任务锁时，两边都会基于同一份旧清单算 plan，后落盘的那个把对方的
    成果覆盖掉（清单说 6 个文档、库里只剩 3 个）。这里用另一个 FileLock 模拟
    "另一个进程正在同步"。
    """
    from rag_assistant.sync.lock import FileLock

    other = FileLock(kb._task_lock.path, stale_seconds=120)
    with other.hold(timeout=0.0):
        with pytest.raises(IndexBusyError, match="同类任务"):
            kb.sync()

    # 对方释放后必须能正常同步，说明锁没有泄漏
    report = kb.sync()
    assert report.failed == []
    assert not kb._task_lock.path.exists()


def test_task_lock_touch_keeps_long_sync_alive(kb: KnowledgeBase) -> None:
    """同步过程中刷时间戳：否则超过 stale_lock_s 会被当成崩溃残留清掉。"""
    import os

    lock = kb._task_lock
    with lock.hold(timeout=0.0):
        stale = time.time() - lock._stale_seconds - 5
        os.utime(lock.path, (stale, stale))
        assert lock.is_locked() is False
        assert lock.touch() is True
        assert lock.is_locked() is True


def test_sync_uses_write_lock(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    """统计写锁进入次数：至少覆盖"删除/提交/落盘"三个临界区。"""
    enters = 0
    original = kb.lock.write

    from contextlib import contextmanager

    @contextmanager
    def counting_write(timeout: float | None = None):
        nonlocal enters
        enters += 1
        with original(timeout=timeout):
            yield

    monkeypatch.setattr(kb.lock, "write", counting_write)
    report = kb.sync()

    assert report.ok
    assert enters >= 2, f"写锁进入次数过少：{enters}"


def test_lock_file_cleaned_after_sync(kb: KnowledgeBase) -> None:
    kb.sync()
    lock_path = kb.settings.resolve(kb.settings.sync.lock_file)
    assert not lock_path.exists(), "同步结束后不应残留锁文件"


def test_query_waits_then_fails_when_writer_busy(
    mock_settings: Settings, sample_raw_dir: Path
) -> None:
    """写者持锁期间查询会等待，超时后抛 IndexBusyError（API 会映射成 409）。"""
    settings = replace(
        mock_settings,
        sync=replace(mock_settings.sync, read_lock_timeout_s=0.15, write_lock_timeout_s=1.0),
    )
    kb = KnowledgeBase(settings)
    kb.sync()

    holding = threading.Event()
    release = threading.Event()

    def writer() -> None:
        with kb.lock.write(timeout=2):
            holding.set()
            release.wait(3)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert holding.wait(2)
        started = time.perf_counter()
        with pytest.raises(IndexBusyError):
            kb.search("休学多久")
        waited = time.perf_counter() - started
        assert waited >= 0.1, "应当确实等待过读锁，而不是立即失败"
    finally:
        release.set()
        thread.join(3)

    # 写者释放后应恢复正常
    assert kb.search("休学多久", threshold=0.0).chunks


def test_manifest_rollback_on_fatal_error(
    kb: KnowledgeBase, sample_raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """提交阶段抛严重异常 → 清单回滚到同步前（不把半成品记录进去）。"""
    kb.sync()
    before = load_manifest(kb.settings.manifest_path)
    assert before.doc_count >= 4

    (sample_raw_dir / "新增办法.md").write_text(
        "# 第一章 总则\n\n第一条 新增内容，用于验证回滚。", encoding="utf-8"
    )

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("模拟磁盘故障")

    monkeypatch.setattr(kb, "_commit_batch", boom)
    report = kb.sync()

    assert not report.ok
    assert any(item["stage"] == "commit" for item in report.failed)
    after = load_manifest(kb.settings.manifest_path)
    assert after.doc_count == before.doc_count, "清单应回滚到同步前的状态"
    assert all(entry.rel_path != "新增办法.md" for entry in after.docs.values())


def test_batch_commit_size(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """commit_batch_docs=2 时，4 个文件应分 2 批提交。"""
    settings = replace(mock_settings, sync=replace(mock_settings.sync, commit_batch_docs=2))
    kb = KnowledgeBase(settings)

    batches: list[int] = []
    original = kb._commit_batch

    def spy(batch, modified_ids, report, chunk_counts):  # type: ignore[no-untyped-def]
        batches.append(len(batch))
        return original(batch, modified_ids, report, chunk_counts)

    kb._commit_batch = spy  # type: ignore[method-assign]
    report = kb.sync()

    assert report.ok
    assert all(size <= 2 for size in batches), f"每批不超过 2 个文件，实际 {batches}"
    assert sum(batches) >= 4, f"所有文件都应被提交，实际 {batches}"


def test_status_exposes_lock(kb: KnowledgeBase) -> None:
    status = kb.status()
    assert "lock" in status
    lock = status["lock"]
    assert lock["file_locked"] is False
    assert "readers" in lock and "writer" in lock


def test_new_file_visible_after_scheduled_sync(kb: KnowledgeBase, sample_raw_dir: Path) -> None:
    """新增文件 → 同步 → 立刻可检索（定时任务的语义就是这个）。"""
    kb.sync()
    assert not any(chunk.doc_name == "临时通知.md" for chunk in kb.search("实验室安全", threshold=0.0).chunks)

    (sample_raw_dir / "临时通知.md").write_text(
        "# 实验室安全通知\n\n## 一、适用范围\n\n本通知适用于所有进入实验室的师生，须佩戴防护眼镜。",
        encoding="utf-8",
    )
    report = kb.sync()
    assert report.added == 1

    hits = kb.search("实验室安全 防护眼镜", top_n=3, threshold=0.0)
    assert hits.chunks
    assert hits.chunks[0].doc_name == "临时通知.md"
