"""心跳监视的日志缓冲与滚动：写盘节奏、容量上限、异常兜底。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import heartbeat_server as hb


def make_log(tmp_path: Path, **kwargs) -> hb.SampleLog:
    config = hb.LogConfig(path=tmp_path / "heartbeat.jsonl", **kwargs)
    log = hb.SampleLog(config)
    log.open()
    return log


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_buffers_until_record_limit(tmp_path: Path) -> None:
    """没攒够条数就不落盘，攒够了才写。"""
    log = make_log(tmp_path, flush_records=3, flush_seconds=3600)
    for index in range(2):
        log.add({"n": index})
    assert not log.path.exists()

    log.add({"n": 2})
    lines = read_lines(log.path)
    assert len(lines) == 3
    assert [json.loads(line)["n"] for line in lines] == [0, 1, 2]


def test_buffers_until_interval(tmp_path: Path) -> None:
    """条数没到但时间到了，也要落盘。"""
    log = make_log(tmp_path, flush_records=1000, flush_seconds=0.05)
    log.add({"n": 1})
    assert not log.path.exists()

    time.sleep(0.06)
    log.add({"n": 2})
    assert len(read_lines(log.path)) == 2


def test_rotate_keeps_bounded_space(tmp_path: Path) -> None:
    """写满就滚，历史份数受 keep 限制，总占用不会无限长。"""
    log = make_log(tmp_path, max_bytes=400, keep=2, flush_records=1, flush_seconds=3600)
    for index in range(40):
        log.add({"n": index, "pad": "x" * 80})

    files = log.existing_files()
    assert len(files) <= log.config.keep + 1
    assert log.backup_path(1).exists()
    assert not log.backup_path(3).exists()
    used, _ = log.usage()
    # 单个文件上限 400，最多 3 份，允许一行超出一点
    assert used <= log.config.cap_bytes + 400
    assert log.rotations > 0


def test_rotation_drops_oldest_first(tmp_path: Path) -> None:
    """滚动的顺序：最老的先被删，而不是把新日志删掉。"""
    log = make_log(tmp_path, max_bytes=300, keep=1, flush_records=1, flush_seconds=3600)
    for index in range(12):
        log.add({"n": index, "pad": "y" * 70})

    active = [json.loads(line)["n"] for line in read_lines(log.path)]
    backup = [json.loads(line)["n"] for line in read_lines(log.backup_path(1))]
    assert max(active) == 11              # 当前文件里是最新的
    assert max(backup) < 11               # 历史里是更早的
    assert min(backup) > 0                # 最早的那批已经被删掉


def test_open_prunes_extra_backups(tmp_path: Path) -> None:
    """上次留下的多余备份，启动时清掉。"""
    for index in range(1, 8):
        (tmp_path / f"heartbeat.{index}.jsonl").write_text("{}\n", encoding="utf-8")

    log = make_log(tmp_path, keep=2)
    assert log.backup_path(1).exists()
    assert log.backup_path(2).exists()
    assert not log.backup_path(3).exists()


def test_flush_survives_write_error(tmp_path: Path, monkeypatch) -> None:
    """写盘失败不能把监视器带崩，内容留着下次重试。"""
    log = make_log(tmp_path)
    log.add({"n": 1})

    original_open = Path.open

    def boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if self == log.path:
            raise OSError("disk full")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", boom)
    assert log.flush(force=True) is False
    assert log.errors == 1
    assert len(log.pending) == 1

    monkeypatch.setattr(Path, "open", original_open)
    assert log.flush(force=True) is True
    assert read_lines(log.path)


def test_pending_has_memory_limit(tmp_path: Path, monkeypatch) -> None:
    """盘一直写不进去时，内存里的积压也要有上限。"""
    log = make_log(tmp_path, flush_records=2)
    original_open = Path.open

    def boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if self == log.path:
            raise OSError("disk full")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", boom)
    for index in range(100):
        log.add({"n": index})
        log.flush()
    assert len(log.pending) <= 40 + 2
    assert log.dropped > 0


def test_records_are_jsonl(tmp_path: Path) -> None:
    log = make_log(tmp_path, flush_records=1)
    log.add({"ts": "2026-11-20T23:00:00", "state": "OK", "sample": {"latency_ms": 12}})
    line = read_lines(log.path)[0]
    record = json.loads(line)
    assert record["state"] == "OK"
    assert record["sample"]["latency_ms"] == 12


def test_cap_bytes_counts_active_and_backups(tmp_path: Path) -> None:
    config = hb.LogConfig(path=tmp_path / "heartbeat.jsonl", max_bytes=1024 * 1024, keep=5)
    assert config.cap_bytes == 6 * 1024 * 1024
