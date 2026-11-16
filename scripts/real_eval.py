"""阶段 F 实跑：真实模型建库 + 评测 + 阈值标定，产出 reports/eval-*.md。

在 CPU 上会跑很久（首次加载 bge-m3 与 reranker 各需几十秒，每题两次 LLM 调用）。
建议后台运行，日志加 -u 便于观察::

    python -u scripts\\real_eval.py                 # 默认标签 real
    python -u scripts\\real_eval.py --label real-rerank --judge rule

脚本做的事：
1. 打印当前模型配置（provider / model / reranker on-off / threshold）；
2. `sync --rebuild` 用真实向量重建索引；
3. 调用 `scripts/evaluate.py` 出报告（含阈值扫描）；
4. 把推荐阈值打印出来，并提示如何写回 config/settings.yaml。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实模型建库 + 评测 + 阈值标定")
    parser.add_argument("--label", default="real", help="报告文件名标签")
    parser.add_argument("--judge", default="rule", choices=["rule", "llm", "none"])
    parser.add_argument("--golden", default=str(ROOT / "data" / "eval" / "golden_set.jsonl"))
    parser.add_argument("--skip-sync", action="store_true", help="跳过重建（索引已就绪时）")
    args = parser.parse_args(argv)

    from rag_assistant.config import load_settings

    settings = load_settings(ROOT / "config" / "settings.yaml", root=ROOT)
    log(
        f"模型配置：embedding={settings.embedding.provider}/{settings.embedding.model} "
        f"reranker={'on' if settings.reranker.is_active else 'off'} "
        f"llm={settings.llm.provider}/{settings.llm.model} "
        f"threshold={settings.retrieval.score_threshold}"
    )
    if settings.embedding.is_mock:
        log("警告：当前是 mock 向量（未安装 sentence-transformers 或强制 mock），结果只能验证链路")

    if not args.skip_sync:
        from rag_assistant.kb import KnowledgeBase

        log("开始重建索引（真实向量）……")
        started = time.perf_counter()
        kb = KnowledgeBase(settings)
        try:
            report = kb.sync(rebuild=True)
        finally:
            kb.close()
        log(
            f"索引完成：新增 {report.added} / 变更 {report.modified} / 删除 {report.removed} / "
            f"chunk {report.chunks_written} / 失败 {len(report.failed)} / 耗时 {time.perf_counter() - started:.1f}s"
        )
        if report.failed:
            for item in report.failed:
                log(f"  失败：{item}")

    log("开始评测（含阈值扫描）……")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate.py"),
        "--label",
        args.label,
        "--judge",
        args.judge,
        "--golden",
        args.golden,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(ROOT))
    log(f"评测结束：退出码 {completed.returncode}，耗时 {time.perf_counter() - started:.1f}s")
    reports = sorted((ROOT / "reports").glob("*.md"))
    if reports:
        log(f"最新报告：{reports[-1]}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
