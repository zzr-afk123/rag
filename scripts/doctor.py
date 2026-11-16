"""一键自检（doctor）：定位"127.0.0.1 拒绝连接"这类启动问题。

用法::

    doctor.bat                 （推荐，自动用 .venv 里的 python）
    python scripts\\doctor.py
    python scripts\\doctor.py --port 8001

检查项：Python 版本 / 项目结构 / 依赖 / 包可导入 / 索引 / 端口与 /health /
模型缓存 / .env 与 Key / 防火墙规则 / 局域网地址。每项给出 PASS/WARN/FAIL 与修复建议。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED = ["yaml", "dotenv", "jieba", "rank_bm25", "chromadb", "fastapi", "uvicorn", "pypdf", "docx", "openpyxl", "tiktoken"]
OPTIONAL = ["sentence_transformers", "torch", "qrcode", "apscheduler", "watchdog", "langchain_openai"]

_results: list[tuple[str, str, str]] = []  # (level, title, detail)


def add(level: str, title: str, detail: str) -> None:
    _results.append((level, title, detail))


def check_python() -> None:
    version = sys.version_info
    ok = version >= (3, 10)
    add(
        "PASS" if ok else "FAIL",
        "Python 版本",
        f"{version.major}.{version.minor}.{version.micro}（需要 >= 3.10）"
        + ("" if ok else "；请安装 Python 3.10+"),
    )


def check_layout() -> None:
    missing = [
        name
        for name in ("config/settings.yaml", "src/rag_assistant", "requirements.txt")
        if not (ROOT / name).exists()
    ]
    add(
        "PASS" if not missing else "FAIL",
        "项目结构",
        "OK" if not missing else f"缺少：{', '.join(missing)}（请在项目根目录运行）",
    )


def check_deps() -> None:
    import importlib

    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            missing.append(name)
    add(
        "PASS" if not missing else "FAIL",
        "必需依赖",
        "全部已安装" if not missing else f"缺少 {missing}；执行：pip install -r requirements.txt",
    )

    optional_missing = []
    for name in OPTIONAL:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            optional_missing.append(name)
    add(
        "PASS" if not optional_missing else "WARN",
        "可选依赖",
        "全部已安装"
        if not optional_missing
        else f"缺少 {optional_missing}；未装 sentence_transformers/torch 时会自动走 mock 模式",
    )


def check_import() -> None:
    try:
        import rag_assistant  # noqa: F401
        from rag_assistant.config import load_settings

        settings = load_settings()
        mode = "mock" if settings.embedding.is_mock else "real"
        add(
            "PASS",
            "包可导入",
            f"rag_assistant v{rag_assistant.__version__}；embedding={settings.embedding.provider}({mode}) "
            f"reranker={'on' if settings.reranker.is_active else 'off'} "
            f"threshold={settings.retrieval.score_threshold}",
        )
    except Exception as exc:  # noqa: BLE001
        add(
            "FAIL",
            "包可导入",
            f"{type(exc).__name__}: {exc}；执行 pip install -e . "
            f"（或确保 PYTHONPATH 含 src 目录）",
        )


def check_index() -> None:
    manifest = ROOT / "data" / "index" / "manifest.json"
    if not manifest.exists():
        add("WARN", "索引", "还没有索引：先运行 python scripts\\sync.py --rebuild（或页面点「开始构建知识库」）")
        return
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        docs = payload.get("docs") or {}
        chunks = sum(int(entry.get("chunks", 0) or 0) for entry in docs.values())
        add("PASS" if chunks else "WARN", "索引", f"{len(docs)} 个文档 / {chunks} 个 chunk（{payload.get('updated_at', '-')}）")
    except Exception as exc:  # noqa: BLE001
        add("FAIL", "索引", f"manifest.json 解析失败：{exc}；删除 data/index 后重新 sync")


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def check_port(port: int) -> None:
    if not _port_open(port):
        add(
            "FAIL",
            f"端口 {port}",
            "没有服务在监听 → 这就是\"拒绝连接\"的原因。启动：start.bat；"
            "若启动窗口已报错，先看报错或运行本脚本的依赖检查",
        )
        return
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        add(
            "PASS",
            f"端口 {port}",
            f"/health 正常：chunks={payload.get('chunks')} llm={payload.get('llm')}"
            f"(mock={payload.get('llm_mock')}) embedder={payload.get('embedder')}"
            f"(mock={payload.get('embedder_mock')})",
        )
    except Exception as exc:  # noqa: BLE001
        add("WARN", f"端口 {port}", f"端口被占用但不是本服务（{type(exc).__name__}）；换端口：start.bat --port 8001")


def _weight_size_mb(directory: Path, min_mb: float = 50.0) -> float | None:
    """返回最大权重文件的 MB 数；目录不存在返回 None，存在但没有像样权重返回 0。"""
    if not directory.exists():
        return None
    best = 0.0
    for item in directory.rglob("*"):
        if item.is_file() and item.suffix in {".safetensors", ".bin"}:
            best = max(best, item.stat().st_size / 1024 / 1024)
    return best if best >= min_mb else 0.0


def check_models() -> None:
    cache = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".cache" / "huggingface" / "hub"
    bge = _weight_size_mb(cache / "models--BAAI--bge-m3")
    rerank = _weight_size_mb(cache / "models--BAAI--bge-reranker-v2-m3")

    if bge is None:
        add("WARN", "模型缓存", "未找到 bge-m3 → 会走 mock 向量。下载：python scripts\\download_models.py BAAI/bge-m3")
        return
    if not bge:
        add(
            "FAIL",
            "模型缓存",
            "bge-m3 目录存在但权重文件不完整（上次下载可能中断）→"
            " 删除该目录或重跑 python scripts\\download_models.py BAAI/bge-m3 继续下载",
        )
        return

    if rerank and rerank > 0:
        detail = f"bge-m3 {bge:.0f} MB；reranker {rerank:.0f} MB"
        add("PASS", "模型缓存", detail)
    else:
        detail = f"bge-m3 {bge:.0f} MB；reranker 未就绪（目录存在但无完整权重，或未下载）"
        detail += " → 设 RAG_RERANKER__ENABLED=false，或 python scripts\\download_models.py BAAI/bge-reranker-v2-m3"
        add("WARN", "模型缓存", detail)


def check_env() -> None:
    env_file = ROOT / ".env"
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        add("PASS", "DeepSeek Key", f"已配置（{key[:4]}****{key[-2:]}）")
        return
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8", errors="replace")
        if "DEEPSEEK_API_KEY=" in content and len(content.split("DEEPSEEK_API_KEY=", 1)[1].splitlines()[0].strip()) > 8:
            add("WARN", "DeepSeek Key", ".env 里有值但当前进程没读到（重启终端或 start.bat）")
            return
    add("WARN", "DeepSeek Key", ".env 未配置 → 自动 mock LLM，回答是规则占位")


def check_firewall(port: int) -> None:
    if sys.platform != "win32":
        return
    try:
        result = subprocess.run(
            ["netsh", "advfirewall", "firewall", "show", "rule", "name=RAG " + str(port)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        text = (result.stdout or "") + (result.stderr or "")
        if "RAG " + str(port) in text and "action=allow" in text.replace(" ", "").lower():
            add("PASS", "防火墙", f"已存在放行规则 RAG {port}")
        else:
            add(
                "WARN",
                "防火墙",
                f"未发现放行规则；局域网/手机访问前用管理员 PowerShell 执行：\n"
                f'      netsh advfirewall firewall add rule name="RAG {port}" dir=in action=allow protocol=TCP localport={port}',
            )
    except Exception:  # noqa: BLE001
        add("WARN", "防火墙", "无法查询（需要管理员权限）；局域网访问不通时请手动放行端口")


def check_lan(port: int) -> None:
    try:
        from rag_assistant.api.netinfo import lan_ipv4_addresses

        ips = lan_ipv4_addresses()
    except Exception as exc:  # noqa: BLE001
        add("WARN", "局域网地址", f"探测失败：{exc}")
        return
    add(
        "PASS" if ips else "WARN",
        "局域网地址",
        "、".join(f"http://{ip}:{port}/" for ip in ips) if ips else "未探测到局域网 IP（只有本机可用）",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 项目自检")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    port = args.port

    for check in (check_python, check_layout, check_deps, check_import, check_index):
        check()
    check_port(port)
    check_models()
    check_env()
    check_firewall(port)
    check_lan(port)

    print("\n===== 自检结果 =====")
    for level, title, detail in _results:
        print(f"[{level:4}] {title}: {detail}")

    fails = sum(1 for level, _, _ in _results if level == "FAIL")
    warns = sum(1 for level, _, _ in _results if level == "WARN")
    print(f"\n汇总：{len(_results)} 项，FAIL {fails}，WARN {warns}")
    if fails:
        print("结论：有阻塞项，按上面 FAIL 的提示修；修完重跑 doctor.bat。")
    elif warns:
        print("结论：能启动，但存在降级项（见 WARN）。")
    else:
        print("结论：全部正常。")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
