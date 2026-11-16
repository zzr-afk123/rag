"""并发原语测试：读写互斥、写者优先、跨进程文件锁、超时。"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from rag_assistant.exceptions import IndexBusyError
from rag_assistant.sync.lock import FileLock, IndexLock, LockOptions, RwLock


# --------------------------------------------------------------------------
# 进程内读写锁
# --------------------------------------------------------------------------
def test_readers_can_run_concurrently() -> None:
    lock = RwLock()
    active = 0
    peak = 0
    guard = threading.Lock()
    start = threading.Event()

    def reader() -> None:
        nonlocal active, peak
        start.wait(1)
        with lock.read(timeout=2):
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.08)
            with guard:
                active -= 1

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(3)

    assert peak >= 2, "多个读锁应能同时持有"


def test_writer_is_exclusive() -> None:
    lock = RwLock()
    holding = threading.Event()
    release = threading.Event()

    def writer() -> None:
        with lock.write(timeout=2):
            holding.set()
            release.wait(2)

    thread = threading.Thread(target=writer)
    thread.start()
    assert holding.wait(2)
    with pytest.raises(IndexBusyError):
        with lock.read(timeout=0.1):
            pass  # pragma: no cover - 不应该进入
    release.set()
    thread.join(3)


def test_write_waits_for_reader_release() -> None:
    lock = RwLock()
    holding = threading.Event()
    release = threading.Event()

    def reader() -> None:
        with lock.read(timeout=2):
            holding.set()
            release.wait(2)

    thread = threading.Thread(target=reader)
    thread.start()
    assert holding.wait(2)
    with pytest.raises(IndexBusyError):
        with lock.write(timeout=0.1):
            pass  # pragma: no cover
    release.set()
    thread.join(3)


def test_writer_priority_blocks_new_readers() -> None:
    """有写者在排队时，新读者要让路（否则写者会饿死）。"""
    lock = RwLock()
    reader_holding = threading.Event()
    release_reader = threading.Event()
    writer_started = threading.Event()

    def first_reader() -> None:
        with lock.read(timeout=2):
            reader_holding.set()
            release_reader.wait(2)

    def writer() -> None:
        writer_started.set()
        with lock.write(timeout=3):
            time.sleep(0.05)

    thread_reader = threading.Thread(target=first_reader)
    thread_reader.start()
    assert reader_holding.wait(2)

    thread_writer = threading.Thread(target=writer)
    thread_writer.start()
    assert writer_started.wait(1)
    time.sleep(0.05)  # 让写者进入 waiting 状态

    with pytest.raises(IndexBusyError):
        with lock.read(timeout=0.1):
            pass  # pragma: no cover

    release_reader.set()
    thread_reader.join(3)
    thread_writer.join(3)


def test_status_reports_counts() -> None:
    lock = RwLock()
    with lock.read():
        status = lock.status()
        assert status["readers"] == 1
        assert status["writer"] is False
    assert lock.status()["readers"] == 0


# --------------------------------------------------------------------------
# 跨进程文件锁
# --------------------------------------------------------------------------
def test_filelock_exclusive(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    first = FileLock(path)
    second = FileLock(path)
    assert first.acquire(timeout=0) is True
    assert second.acquire(timeout=0.1) is False
    first.release()
    assert second.acquire(timeout=0.5) is True
    second.release()
    assert not path.exists()


def test_filelock_stale_is_reclaimed(tmp_path: Path) -> None:
    """进程崩溃留下的锁文件超过阈值后应被自动清理，否则永远无法同步。"""
    path = tmp_path / ".lock"
    path.write_text("{}", encoding="utf-8")
    old = time.time() - 3600
    os.utime(path, (old, old))

    lock = FileLock(path, stale_seconds=60)
    assert lock.is_locked() is False
    assert lock.acquire(timeout=0.5) is True
    lock.release()


def test_filelock_wait_free(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    holder = FileLock(path)
    holder.acquire()

    waiter = FileLock(path)
    assert waiter.wait_free(0.1) is False
    holder.release()
    assert waiter.wait_free(1.0) is True


def test_filelock_info(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    lock = FileLock(path)
    assert lock.info() is None
    lock.acquire()
    info = lock.info()
    assert info is not None and info.get("pid") == os.getpid()
    lock.release()


# --------------------------------------------------------------------------
# IndexLock 门面
# --------------------------------------------------------------------------
def test_indexlock_write_releases_file(tmp_path: Path) -> None:
    lock = IndexLock(tmp_path / ".lock")
    with lock.write(timeout=1):
        assert (tmp_path / ".lock").exists()
    assert not (tmp_path / ".lock").exists()


def test_indexlock_read_blocked_by_external_writer(tmp_path: Path) -> None:
    """另一个进程持有文件锁时，本进程读也要排队（否则会读到半成品索引）。"""
    path = tmp_path / ".lock"
    external = FileLock(path)
    external.acquire()

    lock = IndexLock(path, LockOptions(read_timeout=0.15, write_timeout=0.15))
    with pytest.raises(IndexBusyError, match="另一个进程"):
        with lock.read():
            pass  # pragma: no cover
    external.release()

    with lock.read(timeout=1):
        pass


def test_indexlock_write_blocked_by_external_writer(tmp_path: Path) -> None:
    path = tmp_path / ".lock"
    external = FileLock(path)
    external.acquire()
    lock = IndexLock(path, LockOptions(write_timeout=0.15))
    with pytest.raises(IndexBusyError, match="另一个进程"):
        with lock.write():
            pass  # pragma: no cover
    external.release()


def test_indexlock_status(tmp_path: Path) -> None:
    lock = IndexLock(tmp_path / ".lock")
    status = lock.status()
    for key in ("readers", "writer", "file_locked", "lock_file", "holder"):
        assert key in status
    assert status["file_locked"] is False
