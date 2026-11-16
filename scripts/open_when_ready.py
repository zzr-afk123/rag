"""等 /health 返回 200 再打开浏览器。

为什么需要它：以前是"睡 3 秒就开浏览器"，但真实模型首次加载要 20–60 秒，
用户看到的就是 **ERR_CONNECTION_REFUSED**，而且页面不会自己刷新。

用法::

    python scripts/open_when_ready.py --port 8000
    python scripts/open_when_ready.py --port 8000 --no-browser   # 只等不打开
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request
import webbrowser


def health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="等服务器就绪后打开浏览器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}/"
    health = f"{base}health"
    # 用首页文件的 mtime 当版本号：每次改了页面，打开的 URL 就变一次，
    # 绕过浏览器对 "/" 的旧缓存（这正是"改了还是不显示"的根因）。
    page = Path(__file__).resolve().parents[1] / "src" / "rag_assistant" / "web" / "index.html"
    try:
        version = str(int(page.stat().st_mtime))
    except OSError:  # pragma: no cover
        version = "1"
    open_url = f"{base}?v={version}"
    deadline = time.time() + args.timeout

    while time.time() < deadline:
        if health_ok(health):
            print(f"[ready] {open_url}", flush=True)
            if not args.no_browser:
                webbrowser.open(open_url)
            return 0
        time.sleep(0.5)

    print(
        f"[timeout] {args.timeout:.0f}s 内 /health 未就绪。\n"
        f"          请查看服务窗口的报错，或运行 doctor.bat 自检。\n"
        f"          就绪后可手动打开：{open_url}",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
