"""目录监听（watchdog，可选）。

与定时任务的取舍：
- **定时**：实现简单、行为可预测，最坏延迟 = 一个间隔（默认 60s）；
- **监听**：延迟低（秒级），但文件"正在写入"时会连续触发，必须做防抖，
  而且网络盘/容器挂载卷上的 inotify 不一定可靠。

因此默认用定时，watchdog 作为可选加速项（``sync.enable_watcher=true``）。
两者可以同时开：监听负责快速响应，定时负责兜底（防止漏事件）。

防抖策略：任意事件都重置一个 ``debounce_s`` 的定时器，只有静默期满才真正触发，
避免"复制一个 200MB 的 PDF"过程中触发十几次同步。
"""

from __future__ import annotations

import fnmatch
import threading
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..logging_setup import get_logger

logger = get_logger("sync.watcher")


class RawDirWatcher:
    """监听知识库目录，变更后防抖触发回调。"""

    def __init__(
        self,
        settings: Settings,
        on_change: Callable[[str], None],
        *,
        debounce_s: float | None = None,
    ) -> None:
        self._settings = settings
        self._on_change = on_change
        self._debounce = settings.sync.watcher_debounce_s if debounce_s is None else debounce_s
        self._observer = None
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._events = 0
        self._triggers = 0

    # ---- 生命周期 ----
    @property
    def active(self) -> bool:
        return self._observer is not None

    def start(self) -> bool:
        """启动监听；watchdog 未安装或目录不存在时返回 False（不阻塞服务启动）。"""
        if not self._settings.sync.enable_watcher:
            return False
        if self._observer is not None:
            return True

        directory = self._settings.raw_dir
        if not directory.exists():
            logger.warning("监听目录不存在，跳过 watchdog：%s", directory)
            return False

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("未安装 watchdog，跳过目录监听（pip install watchdog）")
            return False

        outer = self

        class _Handler(FileSystemEventHandler):  # type: ignore[misc, valid-type]
            """任意文件事件都只是"重置防抖定时器"，真正的判断在 notify 里。"""

            def on_any_event(self, event) -> None:  # noqa: ANN001
                if getattr(event, "is_directory", False):
                    return
                path = getattr(event, "dest_path", None) or getattr(event, "src_path", "")
                if path:
                    outer.notify(Path(path))

        observer = Observer()
        observer.schedule(_Handler(), str(directory), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        logger.info("已开始监听 %s（防抖 %.1fs）", directory, self._debounce)
        return True

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=3)
            self._observer = None
            logger.info("目录监听已停止")

    # ---- 事件处理 ----
    def _accept(self, path: Path) -> bool:
        """过滤：只关心白名单扩展名，忽略临时文件与目录。"""
        if path.name.startswith(("~$", ".")):
            return False
        return any(
            fnmatch.fnmatch(path.name.lower(), pattern.lower())
            for pattern in self._settings.sync.include_patterns
        )

    def notify(self, path: Path) -> None:
        """由 handler 调用：重置防抖定时器。"""
        if not self._accept(path):
            return
        self._events += 1
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._debounce, self._fire, args=(path.name,))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def _fire(self, last_name: str) -> None:
        with self._lock:
            self._timer = None
        self._triggers += 1
        logger.info("目录变更（最近文件：%s），触发同步", last_name)
        try:
            self._on_change(f"watchdog:{last_name}")
        except Exception:  # noqa: BLE001 - 回调失败不能把监听线程带崩
            logger.exception("目录变更回调失败")

    def status(self) -> dict[str, object]:
        return {
            "enabled": self._settings.sync.enable_watcher,
            "active": self.active,
            "debounce_s": self._debounce,
            "events": self._events,
            "triggers": self._triggers,
        }


def build_watcher(
    settings: Settings,
    on_change: Callable[[str], None],
) -> RawDirWatcher:
    return RawDirWatcher(settings, on_change)
