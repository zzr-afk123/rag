# -*- coding: utf-8 -*-
"""把心跳屏固件（MicroPython）刷到 ESP32-C3 上。

用法：
    python scripts/upload_board.py                # 自动找串口
    python scripts/upload_board.py --port COM5    # 指定串口
    python scripts/upload_board.py --check        # 只列板子上的文件

为什么要单独写一个脚本：`mpremote fs cat` 会发 Ctrl-C 打断正在跑的程序，
刷完必须 reset，否则板子停在 REPL、既不广播也不响应 BLE。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "hardware" / "esp32c3_heartbeat" / "micropython"
FILES = ("main.py", "ssd1306.py")

_ESP_PORT = re.compile(r"USB\\VID_303A&PID_1001", re.IGNORECASE)


def find_port() -> str | None:
    """按硬件 ID 找 Espressif 原生 USB 串口（VID 303A / PID 1001）。"""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                "Get-PnpDevice -Class Ports -PresentOnly | "
                "ForEach-Object { $_.InstanceId + '|' + $_.FriendlyName }",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001 - 找不到就退化成让用户手填
        return None
    for line in out.splitlines():
        if not _ESP_PORT.search(line):
            continue
        match = re.search(r"\((COM\d+)\)", line)
        if match:
            return match.group(1)
    return None


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("  > " + " ".join(args))
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if check and result.returncode != 0:
        print("    stderr:", (result.stderr or "").strip()[:400])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="刷心跳屏固件到 ESP32-C3")
    parser.add_argument("--port", default="", help="串口，如 COM5；不填则自动找")
    parser.add_argument("--check", action="store_true", help="只列板子上的文件")
    args = parser.parse_args()

    python = sys.executable
    port = args.port or find_port()
    if not port:
        print("[X] 没找到 ESP32-C3 的串口。")
        print("    确认板子插了 USB 数据线（不是只供电的线），再看设备管理器里的 COM 号，")
        print("    然后：python scripts\\upload_board.py --port COM5")
        return 2
    print(f"[1/4] 串口：{port}")

    missing = [name for name in FILES if not (BOARD_DIR / name).exists()]
    if missing:
        print("[X] 固件文件缺失：", ", ".join(missing))
        return 2

    print("[2/4] 板子上的文件：")
    listing = run([python, "-m", "mpremote", "connect", port, "fs", "ls"], check=False)
    print("    " + (listing.stdout or listing.stderr or "").strip().replace("\n", "\n    "))
    if args.check:
        return 0

    print("[3/4] 上传固件（覆盖同名文件）：")
    for name in FILES:
        source = str(BOARD_DIR / name)
        result = run([python, "-m", "mpremote", "connect", port, "fs", "cp", source, f":{name}"])
        if result.returncode != 0:
            print(f"[X] 上传 {name} 失败")
            return 1
        print(f"    {name} -> 板子")

    print("[4/4] 重启板子（让 main.py 重新跑起来）：")
    run([python, "-m", "mpremote", "connect", port, "reset"], check=False)

    print()
    print("[OK] 固件已更新。屏幕右下角应显示 v3；")
    print("     若仍显示旧画面，拔插一次 USB 让它重新启动。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
