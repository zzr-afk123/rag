"""Heartbeat monitor: poll the RAG service and push its state to an ESP32 over BLE.

The board (hardware/esp32c3_heartbeat) is a BLE peripheral exposing the Nordic
UART service. This script is the central: every N seconds it asks the RAG
service how it is doing, then writes one short line to the board, which renders
it on its OLED and beeps when things break.

Why a separate script instead of putting this in the API process: the monitor
has to notice when the API process is *dead*, so it cannot live inside it.

Usage:
    python scripts/heartbeat_server.py                      # monitor :8000, push to board
    python scripts/heartbeat_server.py --no-ble             # no board, just print
    python scripts/heartbeat_server.py --once               # one sample and exit
    python scripts/heartbeat_server.py --probe              # also ask a real question every 5 min

Wire protocol (one line, UTF-8, <= 120 bytes):
    state=OK;lat=132;qpm=4;chunks=90;mock=0;msg=optional
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Nordic UART service, same UUIDs as the sketch
NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

MAX_LINE = 120

#: 仓库根目录（这个脚本在 scripts/ 下），日志默认落在 <repo>/logs/heartbeat/
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "heartbeat" / "heartbeat.jsonl"


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


@dataclass
class Sample:
    ok: bool
    latency_ms: int
    chunks: int = 0
    llm: str = ""
    llm_mock: bool = False
    error: str = ""
    probed: bool = False
    probe_ms: int = 0
    probe_refused: bool = False
    #: 服务端最近 60 秒的真实提问数（/health 提供），拿不到时为 None
    qpm: int | None = None
    online: int = 0


@dataclass
class Monitor:
    """Turns a stream of samples into a state the board understands."""

    fail_after: int = 2
    warn_ms: int = 4000
    probe_warn_ms: int = 15000
    failures: int = 0
    last_ok: float = 0.0
    qpm_window: list[float] = field(default_factory=list)

    def add(self, sample: Sample) -> tuple[str, str, int]:
        """Return (state, message, qpm)."""
        now = time.time()
        self.qpm_window.append(now)
        self.qpm_window = [t for t in self.qpm_window if now - t <= 60.0]
        qpm = sample.qpm if sample.qpm is not None else len(self.qpm_window)

        if not sample.ok:
            self.failures += 1
            if self.failures >= self.fail_after:
                return "DOWN", (sample.error or "health check failed")[:40], qpm
            return "WARN", "first failure, retrying", qpm

        self.failures = 0
        self.last_ok = now
        if sample.llm_mock:
            return "WARN", f"mock LLM: {sample.llm or 'unknown'}"[:40], qpm
        if sample.probed and sample.probe_ms > self.probe_warn_ms:
            return "WARN", f"answer slow {sample.probe_ms}ms"[:40], qpm
        if sample.latency_ms > self.warn_ms:
            return "WARN", f"health slow {sample.latency_ms}ms"[:40], qpm
        return "OK", "", qpm


def build_line(state: str, sample: Sample, qpm: int, message: str) -> str:
    msg = message.replace(";", ",").replace("\n", " ").strip()
    line = (
        f"state={state};lat={sample.latency_ms};qpm={qpm}"
        f";chunks={sample.chunks};mock={1 if sample.llm_mock else 0}"
    )
    if msg:
        line += f";msg={msg}"
    return line[:MAX_LINE]


# --------------------------------------------------------------------------- #
# 日志：内存里攒一会儿再落盘，写满就滚动，总量有硬上限
# --------------------------------------------------------------------------- #
def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


@dataclass
class LogConfig:
    """日志文件的位置与容量预算。"""

    path: Path = DEFAULT_LOG_PATH
    #: 单个文件写满多少字节就滚一份
    max_bytes: int = 2 * 1024 * 1024
    #: 保留几份历史（不含当前这份）
    keep: int = 5
    #: 每多少秒强制落一次盘
    flush_seconds: float = 30.0
    #: 攒够多少条就提前落盘
    flush_records: int = 50

    @property
    def cap_bytes(self) -> int:
        return self.max_bytes * (self.keep + 1)


class SampleLog:
    """心跳样本的缓冲日志。

    这些样本是"每 5 秒一条"的流水，逐条 fsync 既慢又没必要，所以先攒在内存，
    攒够 ``flush_records`` 条或过了 ``flush_seconds`` 秒才写一次盘。

    容量是**硬上限**：当前文件写到 ``max_bytes`` 就改名成 ``xxx.1``，后面的依次
    后移，超出 ``keep`` 的最老一份直接删掉。所以整个目录最多占用
    ``max_bytes * (keep + 1)`` 字节，长期跑也不会把盘吃满。

    写盘失败（比如文件被占用、盘满）不会把监视器带崩：错误记在 ``errors`` 里，
    待写内容继续留着下次再试；积压过多时丢掉最老的一半，保证内存也有上限。
    """

    def __init__(self, config: LogConfig | None = None) -> None:
        self.config = config or LogConfig()
        self.pending: list[str] = []
        self.last_flush = time.monotonic()
        self.written = 0
        self.rotations = 0
        self.dropped = 0
        self.errors = 0

    # -- 路径 ---------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.config.path

    def backup_path(self, index: int) -> Path:
        stem = self.path.stem
        return self.path.with_name(f"{stem}.{index}{self.path.suffix}")

    def existing_files(self) -> list[Path]:
        files = [self.path] + [self.backup_path(i) for i in range(1, self.config.keep + 1)]
        return [p for p in files if p.exists()]

    def usage(self) -> tuple[int, int]:
        """返回 (占用字节数, 文件个数)。"""
        files = self.existing_files()
        return sum(p.stat().st_size for p in files), len(files)

    def describe(self) -> str:
        used, count = self.usage()
        return (
            f"log: {self.path}（当前 {count} 个文件 / {human_bytes(used)}；"
            f"上限 {human_bytes(self.config.max_bytes)} × {self.config.keep + 1} = "
            f"{human_bytes(self.config.cap_bytes)}；每 {int(self.config.flush_seconds)} 秒"
            f"或 {self.config.flush_records} 条落盘）"
        )

    # -- 生命周期 -----------------------------------------------------------
    def open(self) -> None:
        """建目录，顺手清掉超出保留份数的旧文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for index in range(self.config.keep + 1, 64):
            victim = self.backup_path(index)
            if victim.exists():
                try:
                    victim.unlink()
                except OSError:
                    self.errors += 1
        self.last_flush = time.monotonic()

    # -- 写入 ---------------------------------------------------------------
    def add(self, record: dict[str, Any]) -> None:
        self.pending.append(json.dumps(record, ensure_ascii=False))
        if (
            len(self.pending) >= self.config.flush_records
            or (time.monotonic() - self.last_flush) >= self.config.flush_seconds
        ):
            self.flush()

    def flush(self, *, force: bool = False) -> bool:
        """把攒下的记录写盘。写成功返回 True。"""
        if not self.pending:
            self.last_flush = time.monotonic()
            return True
        blob = "\n".join(self.pending) + "\n"
        try:
            self._rotate_if_needed(len(blob.encode("utf-8")))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(blob)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass  # 某些文件系统不支持 fsync，已经 write+flush 就够了
        except OSError:
            self.errors += 1
            limit = self.config.flush_records * 20
            if len(self.pending) > limit:
                # 盘一直写不进去时别把内存也吃光
                keep_from = len(self.pending) - self.config.flush_records * 10
                self.dropped += keep_from
                self.pending = self.pending[keep_from:]
            return False
        self.written += len(self.pending)
        self.pending.clear()
        self.last_flush = time.monotonic()
        return True

    def _rotate_if_needed(self, incoming: int) -> None:
        current = self.path.stat().st_size if self.path.exists() else 0
        if current + incoming <= self.config.max_bytes:
            return
        if self.config.keep <= 0:
            if self.path.exists():
                self.path.unlink()
            self.rotations += 1
            return
        oldest = self.backup_path(self.config.keep)
        if oldest.exists():
            try:
                oldest.unlink()
            except OSError:
                self.errors += 1
        for index in range(self.config.keep - 1, 0, -1):
            source = self.backup_path(index)
            if source.exists():
                try:
                    source.replace(self.backup_path(index + 1))
                except OSError:
                    self.errors += 1
        if self.path.exists():
            try:
                self.path.replace(self.backup_path(1))
            except OSError:
                self.errors += 1
        self.rotations += 1


# --------------------------------------------------------------------------- #
# talking to the RAG service
# --------------------------------------------------------------------------- #
class RagProbe:
    def __init__(self, base_url: str, timeout: float = 4.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._guest = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health(self) -> Sample:
        started = time.perf_counter()
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            latency = int((time.perf_counter() - started) * 1000)
            if resp.status_code != 200:
                return Sample(False, latency, error=f"HTTP {resp.status_code}")
            data = resp.json()
            raw_qpm = data.get("questions_per_minute")
            return Sample(
                True,
                latency,
                chunks=int(data.get("chunks") or data.get("vector_chunks") or 0),
                llm=str(data.get("llm") or ""),
                llm_mock=bool(data.get("llm_mock")),
                qpm=int(raw_qpm) if isinstance(raw_qpm, (int, float)) else None,
                online=int(data.get("online") or 0),
            )
        except Exception as exc:  # connection refused, timeout, bad JSON
            latency = int((time.perf_counter() - started) * 1000)
            return Sample(False, latency, error=f"{type(exc).__name__}")

    async def probe(self, question: str) -> Sample:
        """Ask one real question - proves the model answers, not just that the port is open."""
        try:
            client = await self._get_client()
            if not self._guest:
                try:
                    await client.post("/auth/guest")
                except Exception:
                    pass
                self._guest = True
            started = time.perf_counter()
            resp = await client.post(
                "/chat",
                json={"question": question, "session_id": "heartbeat"},
                timeout=max(self.timeout * 8, 30.0),
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            if resp.status_code != 200:
                return Sample(False, elapsed, error=f"chat HTTP {resp.status_code}", probed=True)
            data: dict[str, Any] = resp.json()
            return Sample(
                True,
                0,
                probed=True,
                probe_ms=int(data.get("latency_ms") or elapsed),
                probe_refused=bool(data.get("refused")),
            )
        except Exception as exc:
            return Sample(False, 0, error=f"probe {type(exc).__name__}", probed=True)


# --------------------------------------------------------------------------- #
# talking to the board
# --------------------------------------------------------------------------- #
class Board:
    """Keeps a BLE connection to the ESP32 alive and writes status lines."""

    def __init__(self, name: str, scan_timeout: float = 10.0, verbose: bool = False) -> None:
        self.name = name
        self.scan_timeout = scan_timeout
        self.verbose = verbose
        self._client: Any = None
        self._bleak: Any = None

    @property
    def connected(self) -> bool:
        return bool(self._client and self._client.is_connected)

    async def _load_bleak(self) -> bool:
        if self._bleak is not None:
            return True
        try:
            import bleak  # noqa: F401

            self._bleak = bleak
            return True
        except ImportError:
            return False

    async def connect(self) -> bool:
        if self.connected:
            return True
        if not await self._load_bleak():
            print("[ble] bleak not installed: pip install bleak", file=sys.stderr)
            return False
        from bleak import BleakClient, BleakScanner

        try:
            device = await BleakScanner.find_device_by_name(self.name, timeout=self.scan_timeout)
            if device is None:
                if self.verbose:
                    print(f"[ble] device not found: {self.name}", file=sys.stderr)
                return False
            client = BleakClient(device, timeout=20.0)
            await client.connect()
            self._client = client
            try:
                await client.start_notify(NUS_TX_UUID, lambda _c, _d: None)
            except Exception:
                pass  # notify is optional, we only write
            return True
        except Exception as exc:
            if self.verbose:
                print(f"[ble] connect failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
            self._client = None

    async def send(self, line: str) -> bool:
        if not self.connected and not await self.connect():
            return False
        payload = (line.rstrip("\n") + "\n").encode("utf-8")
        try:
            # ATT 默认 MTU 23 → 单个写包只能 20 字节；board 侧按 \n 拼包，这里顺序分片即可
            chunk_size = 20
            for start in range(0, len(payload), chunk_size):
                chunk = payload[start : start + chunk_size]
                await self._client.write_gatt_char(NUS_RX_UUID, chunk, response=True)
            return True
        except Exception as exc:
            if self.verbose:
                print(f"[ble] write failed: {type(exc).__name__}", file=sys.stderr)
            await self.disconnect()
            return False


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
async def run(args: argparse.Namespace) -> int:
    probe = RagProbe(args.url, timeout=args.timeout)
    monitor = Monitor(fail_after=args.fail_after, warn_ms=args.warn_ms)
    board = Board(args.device, scan_timeout=args.scan_timeout, verbose=args.verbose)

    sample_log: SampleLog | None = None
    if not args.no_log:
        target = Path(args.log) if args.log else DEFAULT_LOG_PATH
        if not target.is_absolute():
            target = REPO_ROOT / target
        sample_log = SampleLog(
            LogConfig(
                path=target,
                max_bytes=int(args.log_max_mb * 1024 * 1024),
                keep=args.log_keep,
                flush_seconds=args.log_flush,
                flush_records=args.log_records,
            )
        )
        sample_log.open()
        used, count = sample_log.usage()
        print(f"[{_now()}] {sample_log.describe()}")
        if count:
            print(f"[{_now()}] 已有日志 {count} 个文件 / {human_bytes(used)}，接着往后写")

    print(f"[{_now()}] monitoring {args.url} every {args.interval}s, board={args.device if not args.no_ble else 'none'}")
    if not args.no_ble and not board._bleak:
        await board._load_bleak()
        if board._bleak is None:
            print("[!] bleak missing, exit. install: <venv>\\Scripts\\python.exe -m pip install bleak", file=sys.stderr)
            return 2

    started = time.time()
    last_probe = 0.0
    try:
        while True:
            sample = await probe.health()

            if args.probe and (time.time() - last_probe) >= args.probe_every * 60:
                last_probe = time.time()
                deep = await probe.probe(args.question)
                if deep.ok:
                    sample.probed = True
                    sample.probe_ms = deep.probe_ms
                    sample.probe_refused = deep.probe_refused

            state, message, qpm = monitor.add(sample)
            line = build_line(state, sample, qpm, message)

            ble_state = "off"
            if not args.no_ble:
                ok = await board.send(line)
                ble_state = "sent" if ok else "no-board"

            uptime = int(time.time() - started)
            print(
                f"[{_now()}] {state:<4} lat={sample.latency_ms:>4}ms qpm={qpm:<2} "
                f"chunks={sample.chunks:<3} ble={ble_state} up={uptime}s"
                + (f" msg={message}" if message else "")
            )

            if sample_log is not None:
                sample_log.add(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "state": state,
                        "line": line,
                        "ble": ble_state,
                        "sample": sample.__dict__,
                    }
                )
                # 出问题的那一条不要等到攒满才写：按下去就是证据
                if state != "OK":
                    sample_log.flush()

            if args.once:
                break
            await asyncio.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        if sample_log is not None:
            if sample_log.flush(force=True):
                used, count = sample_log.usage()
                print(
                    f"[{_now()}] 日志已落盘：{sample_log.written} 条 / "
                    f"{count} 个文件 / {human_bytes(used)}（滚动 {sample_log.rotations} 次）"
                )
            else:
                print(f"[{_now()}] 日志写入失败 {sample_log.errors} 次，剩余未写 {len(sample_log.pending)} 条")
        await board.disconnect()
        await probe.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 心跳监视：轮询服务状态，推给 ESP32 板子")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="RAG 服务地址")
    parser.add_argument("--interval", type=float, default=5.0, help="轮询间隔（秒）")
    parser.add_argument("--timeout", type=float, default=4.0, help="/health 超时（秒）")
    parser.add_argument("--fail-after", type=int, default=2, help="连续失败几次判为 DOWN")
    parser.add_argument("--warn-ms", type=int, default=4000, help="/health 超过这个耗时算 WARN")
    parser.add_argument("--device", default="RAG-HB", help="板子的 BLE 名称")
    parser.add_argument("--scan-timeout", type=float, default=10.0, help="扫板子的超时（秒）")
    parser.add_argument("--no-ble", action="store_true", help="不连板子，只在控制台打印")
    parser.add_argument("--once", action="store_true", help="采一次就退出")
    parser.add_argument("--probe", action="store_true", help="额外问一个真问题，验证模型能答")
    parser.add_argument("--probe-every", type=float, default=5.0, help="每隔几分钟问一次真问题")
    parser.add_argument(
        "--question", default="学生休学最长可以多久？", help="探活用的问题"
    )
    parser.add_argument(
        "--log",
        default="",
        help="日志文件路径（默认 logs/heartbeat/heartbeat.jsonl，相对路径按仓库根算）",
    )
    parser.add_argument("--no-log", action="store_true", help="完全不写日志文件")
    parser.add_argument("--log-max-mb", type=float, default=2.0, help="单个日志文件的上限（MB）")
    parser.add_argument("--log-keep", type=int, default=5, help="保留几份历史日志")
    parser.add_argument("--log-flush", type=float, default=30.0, help="每多少秒落一次盘")
    parser.add_argument("--log-records", type=int, default=50, help="攒够多少条就提前落盘")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
