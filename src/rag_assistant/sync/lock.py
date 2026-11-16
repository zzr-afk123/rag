"""索引并发控制：进程内读写锁 + 跨进程文件锁。

动机（需求 6：更新过程需加锁，避免索引重建与查询冲突）：

| 冲突场景 | 后果 | 由谁防住 |
|---|---|---|
| 同步正在 upsert 时查询 | 读到半新半旧的数据、Chroma 内部状态竞争 | 进程内读写锁 |
| 命令行 ``scripts/sync.py`` 与 API 进程同时同步 | 两套流程互相覆盖向量/清单，出现"清单说 4 个文档、库里只有 2 个" | 跨进程文件锁 |
| 同步进程崩溃留下锁文件 | 后续永远无法同步 | 锁文件超时视为残留（``stale_lock_s``） |

读写锁的语义：**写者优先**。否则持续不断的查询会让同步永远等不到写锁（写饥饿）。
代价是"有写者在等"时新查询会排队——所以读锁也要有超时，宁可返回 409 也不无限挂起。

**非重入**：同一个线程不能在持有读锁时再申请写锁（会自己等自己）。
当前代码没有这种嵌套；如果将来出现，请拆成两次独立的临界区。
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..exceptions import IndexBusyError
from ..logging_setup import get_logger

logger = get_logger("sync.lock")

_POLL_INTERVAL = 0.02


class RwLock:
    """进程内读写锁（写者优先）。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    # ---- 内部 ----
    def _acquire_read(self, timeout: float | None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._writer or self._waiting_writers:
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            self._readers += 1
            return True

    def _release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers <= 0:
                self._readers = 0
                self._cond.notify_all()

    def _acquire_write(self, timeout: float | None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    if deadline is None:
                        self._cond.wait()
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(remaining)
                self._writer = True
                return True
            finally:
                self._waiting_writers -= 1
                self._cond.notify_all()

    def _release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    # ---- 对外 ----
    @contextlib.contextmanager
    def read(self, timeout: float | None = None) -> Iterator[None]:
        if not self._acquire_read(timeout):
            raise IndexBusyError("等待读锁超时：有同步任务正在写入索引")
        try:
            yield
        finally:
            self._release_read()

    @contextlib.contextmanager
    def write(self, timeout: float | None = None) -> Iterator[None]:
        if not self._acquire_write(timeout):
            raise IndexBusyError("等待写锁超时：有查询正在读取索引")
        try:
            yield
        finally:
            self._release_write()

    def status(self) -> dict[str, int | bool]:
        with self._cond:
            return {
                "readers": self._readers,
                "writer": self._writer,
                "waiting_writers": self._waiting_writers,
            }


class FileLock:
    """跨进程独占锁（基于 O_CREAT|O_EXCL 的原子创建）。

    不依赖 fcntl/msvcrt，Windows/Linux 行为一致；锁文件里写 pid 与时间戳，
    超过 ``stale_seconds`` 认为是崩溃残留并自动清理。
    """

    def __init__(self, path: Path, *, stale_seconds: float = 900.0) -> None:
        self._path = Path(path)
        self._stale_seconds = stale_seconds
        self._held = False
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def _age(self) -> float:
        try:
            return time.time() - self._path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _is_stale(self) -> bool:
        return self._age() > self._stale_seconds

    def is_locked(self) -> bool:
        """锁文件存在且未过期。"""
        return self._path.exists() and not self._is_stale()

    def acquire(self, timeout: float = 0.0) -> bool:
        """尝试获取锁；``timeout=0`` 表示只试一次。"""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._is_stale():
                    logger.warning("清理过期锁文件（可能上次同步异常退出）：%s", self._path)
                    with contextlib.suppress(FileNotFoundError):
                        self._path.unlink()
                    continue
                if time.monotonic() >= deadline:
                    return False
                time.sleep(_POLL_INTERVAL)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "acquired_at": time.time()}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            self._held = True
            return True

    def touch(self) -> bool:
        """刷新锁文件时间戳（心跳）。

        没有心跳时，超过 ``stale_seconds`` 的长任务会被另一个进程判定为"崩溃残留"，
        把锁删掉再抢走，于是两个进程同时写索引。同步循环里每个文件刷一次即可。
        """
        if not self._held:
            return False
        try:
            os.utime(self._path, None)
            return True
        except FileNotFoundError:
            return False
        except OSError:  # pragma: no cover - 磁盘异常时不应打断同步
            logger.warning("刷新锁文件时间戳失败：%s", self._path, exc_info=True)
            return False

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()

    @contextlib.contextmanager
    def hold(self, timeout: float = 0.0) -> Iterator[None]:
        """持有整段任务期间（不等锁时抛 IndexBusyError）。"""
        if not self.acquire(timeout):
            raise IndexBusyError("另一个进程正在执行同类任务，请稍后再试")
        try:
            yield
        finally:
            self.release()

    def wait_free(self, timeout: float) -> bool:
        """等待锁释放（用于读操作跨进程一致性）。"""
        deadline = time.monotonic() + max(0.0, timeout)
        while self.is_locked():
            if time.monotonic() >= deadline:
                return False
            time.sleep(_POLL_INTERVAL)
        return True

    def info(self) -> dict[str, object] | None:
        """读取锁文件内容（诊断用）。"""
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"corrupt": True}

    def __del__(self) -> None:  # pragma: no cover - 兜底释放
        with contextlib.suppress(Exception):
            self.release()


@dataclass
class LockOptions:
    """锁等待参数。"""

    write_timeout: float = 30.0
    read_timeout: float = 20.0
    stale_seconds: float = 900.0


class IndexLock:
    """索引锁门面：查询走读锁，同步走"文件锁 + 写锁"。"""

    def __init__(self, lock_file: Path, options: LockOptions | None = None) -> None:
        self._options = options or LockOptions()
        self._rw = RwLock()
        self._file = FileLock(lock_file, stale_seconds=self._options.stale_seconds)

    # ---- 临界区 ----
    @contextlib.contextmanager
    def read(self, timeout: float | None = None) -> Iterator[None]:
        """读临界区：先等跨进程的写者结束，再拿进程内读锁。"""
        wait = self._options.read_timeout if timeout is None else timeout
        if not self._file.wait_free(wait):
            raise IndexBusyError("索引正在被另一个进程同步，请稍后重试")
        with self._rw.read(wait):
            yield

    @contextlib.contextmanager
    def write(self, timeout: float | None = None) -> Iterator[None]:
        """写临界区：跨进程独占 + 进程内独占。"""
        wait = self._options.write_timeout if timeout is None else timeout
        if not self._file.acquire(wait):
            raise IndexBusyError("另一个进程正在同步索引，请稍后重试")
        try:
            with self._rw.write(wait):
                yield
        finally:
            self._file.release()

    # ---- 诊断 ----
    def heartbeat(self) -> bool:
        """同步循环里周期性调用，刷新跨进程锁的时间戳。"""
        return self._file.touch()

    def status(self) -> dict[str, object]:
        return {
            **self._rw.status(),
            "file_locked": self._file.is_locked(),
            "lock_file": str(self._file.path),
            "holder": self._file.info(),
        }
