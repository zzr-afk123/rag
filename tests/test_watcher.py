"""目录监听测试：防抖合并、格式过滤、真实 watchdog 事件。"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rag_assistant.config import Settings
from rag_assistant.sync.watcher import RawDirWatcher


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, reason: str) -> None:
        self.calls.append(reason)


@pytest.fixture()
def watch_settings(mock_settings: Settings) -> Settings:
    return replace(
        mock_settings,
        sync=replace(mock_settings.sync, enable_watcher=True, watcher_debounce_s=0.15),
    )


def _wait_for(predicate, timeout: float = 3.0, poll: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def test_disabled_by_default(mock_settings: Settings) -> None:
    watcher = RawDirWatcher(mock_settings, Recorder())
    assert watcher.start() is False
    assert watcher.active is False
    assert watcher.status()["enabled"] is False


def test_debounce_coalesces_events(watch_settings: Settings, raw_dir: Path) -> None:
    """连续多次变更只触发一次同步（复制大文件时不会触发十几次）。"""
    recorder = Recorder()
    watcher = RawDirWatcher(watch_settings, recorder, debounce_s=0.15)
    try:
        for _ in range(5):
            watcher.notify(raw_dir / "学生学籍管理规定.md")
            time.sleep(0.02)
        assert _wait_for(lambda: len(recorder.calls) == 1, timeout=2)
        time.sleep(0.2)
        assert len(recorder.calls) == 1
        assert watcher.status()["events"] >= 5
        assert watcher.status()["triggers"] == 1
    finally:
        watcher.stop()


def test_unsupported_extension_ignored(watch_settings: Settings, raw_dir: Path) -> None:
    recorder = Recorder()
    watcher = RawDirWatcher(watch_settings, recorder, debounce_s=0.05)
    try:
        watcher.notify(raw_dir / "表格.csv")
        watcher.notify(raw_dir / "~$临时.docx")
        time.sleep(0.2)
        assert recorder.calls == []
        assert watcher.status()["events"] == 0
    finally:
        watcher.stop()


def test_real_watchdog_observer(watch_settings: Settings, raw_dir: Path) -> None:
    """真起一个 watchdog observer，写文件后应触发回调。"""
    pytest.importorskip("watchdog")
    recorder = Recorder()
    watcher = RawDirWatcher(watch_settings, recorder, debounce_s=0.2)
    assert watcher.start() is True
    assert watcher.active is True
    try:
        target = raw_dir / "新增制度.md"
        target.write_text("# 第一章 总则\n\n第一条 内容。", encoding="utf-8")
        assert _wait_for(lambda: bool(recorder.calls), timeout=5), "watchdog 未捕获到文件变更"
        assert "新增制度.md" in recorder.calls[-1] or "watchdog:" in recorder.calls[-1]
    finally:
        watcher.stop()
    assert watcher.active is False


def test_stop_is_idempotent(watch_settings: Settings, raw_dir: Path) -> None:
    watcher = RawDirWatcher(watch_settings, Recorder(), debounce_s=0.05)
    watcher.start()
    watcher.stop()
    watcher.stop()  # 不应抛异常
    assert watcher.active is False


def test_missing_directory(watch_settings: Settings, mock_settings: Settings) -> None:
    """目录不存在时返回 False，不能阻塞服务启动。"""
    broken = replace(watch_settings, sync=replace(watch_settings.sync, watch_dir="data/not-exist"))
    watcher = RawDirWatcher(broken, Recorder())
    assert watcher.start() is False


def test_status_shape(watch_settings: Settings) -> None:
    watcher = RawDirWatcher(watch_settings, Recorder())
    status: dict[str, Any] = watcher.status()
    for key in ("enabled", "active", "debounce_s", "events", "triggers"):
        assert key in status
