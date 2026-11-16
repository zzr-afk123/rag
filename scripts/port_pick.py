"""挑一个可用端口（被占用时自动顺延），只打印端口号。

用法::

    python scripts/port_pick.py --host 0.0.0.0 --port 8000
    -> 8000（可用）或 8001/8002…（8000 被占用）

给 ``start.bat`` 用：``for /f %%p in ('python scripts\\port_pick.py ...') do set PORT=%%p``
"""

from __future__ import annotations

import argparse
import socket


def is_free(host: str, port: int) -> bool:
    """能否绑定该端口（Windows 上 0.0.0.0 已被占用时，绑 127.0.0.1 也会失败）。"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.3)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="选择一个可用端口")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tries", type=int, default=20)
    args = parser.parse_args(argv)

    host = "0.0.0.0" if args.host in {"0.0.0.0", "::", ""} else args.host
    for offset in range(args.max_tries):
        candidate = args.port + offset
        if is_free(host, candidate):
            print(candidate)
            return 0
    print(args.port)  # 全被占用时仍返回原端口，让 uvicorn 报明确错误
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
