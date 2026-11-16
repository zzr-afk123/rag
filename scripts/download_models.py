"""下载真实模型到 HuggingFace 缓存（走 hf-mirror 镜像）。

用法::

    python scripts/download_models.py                 # 默认两个模型
    python scripts/download_models.py BAAI/bge-m3      # 只下一个

说明：
- 官方 huggingface.co 在部分网络下不可达，因此默认把 ``HF_ENDPOINT`` 指向
  ``https://hf-mirror.com``；如果你的网络能直连官方站，设 ``HF_ENDPOINT=https://huggingface.co``。
- 优先只要 ``safetensors``（忽略 ``*.bin`` 与 onnx/msgpack 等冗余权重），
  若某仓库没有 safetensors 会自动重下完整权重。
- 模型体积：bge-m3 约 2.3GB、bge-reranker-v2-m3 约 2.3GB。
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 镜像站（以及不少企业代理）不支持 HuggingFace 新的 Xet 传输协议，
# 它会去 cas-server.xethub.hf.co 换票并返回 401；关掉后用普通 HTTP 下载。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

DEFAULT_REPOS = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"]
# 只跳过冗余格式与仓库里的杂项文件（镜像站对 imgs/.DS_Store 这类垃圾文件会返回 403，
# 不排除掉会直接让整个 snapshot 失败）。注意**不要**跳过 *.bin：
# bge-m3 只有 pytorch_model.bin，没有 safetensors。
_SKIP = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.onnx",
    "onnx/*",
    "coreml/*",
    "*.mlpackage",
    "imgs/*",
    "*.DS_Store",
    "*.md",
    "*.png",
    "*.jpg",
    "*.gif",
    "*.gitattributes",
]


def _has_weights(path: str) -> bool:
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".safetensors", ".bin")):
                return True
    return False


def download(repo: str) -> str:
    from huggingface_hub import snapshot_download

    started = time.perf_counter()
    print(f"[下载] {repo}（镜像 {os.environ['HF_ENDPOINT']}）...", flush=True)
    try:
        path = snapshot_download(repo_id=repo, ignore_patterns=_SKIP, max_workers=4)
        if not _has_weights(path):
            print(f"  未发现权重文件，改为完整下载（含 README/图片）：{repo}", flush=True)
            path = snapshot_download(repo_id=repo, max_workers=4)
    except Exception as exc:  # noqa: BLE001
        print(f"[失败] {repo}: {type(exc).__name__}: {exc}", flush=True)
        raise

    size_mb = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(path)
        for name in files
    ) / 1024 / 1024
    print(f"[完成] {repo} -> {path}（{size_mb:.0f} MB，{time.perf_counter() - started:.0f}s）", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    repos = (argv if argv is not None else sys.argv[1:]) or DEFAULT_REPOS
    for repo in repos:
        download(repo)
    print("[全部完成]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
