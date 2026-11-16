"""打印可访问地址（服务器模式用）。可选：终端二维码，手机扫码直达。

用法::

    python scripts/show_urls.py                      # 默认 127.0.0.1:8000
    python scripts/show_urls.py --host 0.0.0.0 --port 8000
    python scripts/show_urls.py --host 0.0.0.0 --no-qr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_assistant.api.netinfo import build_urls  # noqa: E402


def _print_qr(url: str) -> bool:
    """尽量在终端打印二维码；未装 qrcode 时静默跳过。"""
    try:
        import qrcode
    except ImportError:
        return False
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:  # noqa: BLE001 - 终端编码不支持时不要影响启动
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打印本服务可访问的地址")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-qr", action="store_true", help="不打印二维码")
    args = parser.parse_args(argv)

    urls = build_urls(args.host, args.port)
    for url in urls:
        print(f"      {url}")
    if urls:
        print(f"      部门看板：{urls[0]}dept")

    if args.host in {"0.0.0.0", "::", ""}:
        print("      （以上为局域网地址，手机需连同一个 Wi-Fi）")
        if not args.no_qr and urls:
            print()
            if not _print_qr(urls[0]):
                print("      （未安装 qrcode，无法打印二维码；pip install qrcode 可开启）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
