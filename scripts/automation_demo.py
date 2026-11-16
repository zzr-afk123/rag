"""阶段 E 验收脚本：验证"新增文档在可配置时间内可被检索到"。

在**临时目录**里跑，不碰仓库的 data/：

1. 复制 config 与示例文档到临时项目根；
2. 全量建索引；
3. 启动定时同步（默认 5s，可调）；
4. 写入一份新文档，轮询检索接口，记录"从落盘到可检索"的耗时；
5. 关闭调度器并清理临时目录。

用法::

    python scripts/automation_demo.py                 # 定时同步（默认 5s 间隔）
    python scripts/automation_demo.py --interval 60   # 生产默认 60s
    python scripts/automation_demo.py --watcher       # 同时打开 watchdog 监听
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

from rag_assistant.api.jobs import SyncJobManager  # noqa: E402
from rag_assistant.config import load_settings  # noqa: E402
from rag_assistant.kb import KnowledgeBase  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402
from rag_assistant.sync.scheduler import SyncScheduler  # noqa: E402
from rag_assistant.sync.watcher import RawDirWatcher  # noqa: E402

NEW_DOC = """# 实验室安全管理规定

## 第一章 总则

**第一条 适用范围**

本规定适用于所有进入实验室的师生，进入实验区必须佩戴防护眼镜并穿实验服。

**第二条 培训要求**

首次进入实验室的人员须完成不少于 4 学时的安全培训，考核合格后方可独立操作。
"""

PROBE_QUESTION = "进入实验室需要佩戴什么，培训多少学时"


def _prepare_project(root: Path, interval: int, watcher: bool) -> Path:
    """在临时目录搭一份可运行的项目骨架。"""
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    for name in ("settings.yaml", "logging.yaml", "jieba_userdict.txt", "stopwords_zh.txt"):
        shutil.copy(PROJECT_ROOT / "config" / name, root / "config" / name)

    for item in (PROJECT_ROOT / "data" / "raw").iterdir():
        if item.is_file() and item.suffix.lower() in {".md", ".txt"}:
            shutil.copy(item, root / "data" / "raw" / item.name)

    config_path = root / "config" / "settings.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["sync"].update(
        {
            "mode": "schedule",
            "interval_seconds": interval,
            "enable_watcher": watcher,
            "watcher_debounce_s": 1.0,
        }
    )
    data["embedding"].update({"provider": "mock", "model": "mock"})
    data["reranker"].update({"provider": "mock"})
    data["llm"].update({"provider": "mock"})
    data["vectorstore"]["persist_dir"] = "data/index/chroma"
    data["logging"]["dir"] = "logs"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证新增文档的可检索延迟")
    parser.add_argument("--interval", type=int, default=5, help="定时同步间隔（秒）")
    parser.add_argument("--watcher", action="store_true", help="同时启用 watchdog")
    parser.add_argument("--timeout", type=int, default=90, help="最长等待时间（秒）")
    args = parser.parse_args(argv)

    temp_root = Path(tempfile.mkdtemp(prefix="rag-automation-"))
    try:
        config_path = _prepare_project(temp_root, args.interval, args.watcher)
        settings = load_settings(config_path, root=temp_root, env={})
        setup_logging(settings, force=True)

        kb = KnowledgeBase(settings)
        initial = kb.sync(rebuild=True)
        print(f"[1] 初始建库：{initial.summary()}")
        print(f"    已入库 chunk={kb.vector_store.count()}，文档={kb.status()['documents']}")

        jobs = SyncJobManager(kb)
        scheduler = SyncScheduler(settings, jobs)
        scheduler.start()
        watcher = RawDirWatcher(settings, lambda reason: jobs.trigger(rebuild=False))
        watcher.start()
        print(f"[2] 自动化已启动：定时={scheduler.status()}  监听={watcher.status()['active']}")

        before = kb.search(PROBE_QUESTION, threshold=0.0)
        print(f"[3] 新增前检索：返回 {len(before.chunks)} 条，命中实验室文档={any('实验室' in c.doc_name for c in before.chunks)}")

        target = settings.raw_dir / "实验室安全管理规定.md"
        started = time.perf_counter()
        target.write_text(NEW_DOC, encoding="utf-8")
        print(f"[4] 已写入新文档：{target.name}")

        found = False
        elapsed = 0.0
        while time.perf_counter() - started < args.timeout:
            time.sleep(0.3)
            hits = kb.search(PROBE_QUESTION, top_n=3, threshold=0.0)
            if any(chunk.doc_name == target.name for chunk in hits.chunks):
                found = True
                elapsed = time.perf_counter() - started
                break

        if found:
            print(f"[5] 可检索耗时：{elapsed:.1f}s（配置间隔 {args.interval}s，监听={'on' if args.watcher else 'off'}）")
            print(f"    Top-1: {kb.search(PROBE_QUESTION, top_n=1, threshold=0.0).chunks[0].section_path}")
        else:
            print(f"[5] 超时未检索到（等待 {args.timeout}s）")

        status = kb.status()
        print(f"[6] 索引状态：文档 {status['documents']} / 向量 {status['vector_chunks']} / BM25 {status['bm25_chunks']}")
        print(f"    锁状态：{status['lock']}")

        watcher.stop()
        scheduler.shutdown()
        kb.close()
        return 0 if found else 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
