"""知识库同步 CLI（阶段 B 可用）。

用法::

    python scripts/sync.py                 # 增量同步
    python scripts/sync.py --rebuild       # 清空后全量重建
    python scripts/sync.py --status        # 只看状态，不同步
    python scripts/sync.py --limit 2       # 只处理前 2 个待同步文件
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_assistant.config import get_settings  # noqa: E402
from rag_assistant.kb import KnowledgeBase  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402


def main(argv: list[str] | None = None, *, settings=None) -> int:
    """CLI 入口。

    ``settings`` 允许注入（测试用临时项目目录），缺省读真实配置。
    """
    parser = argparse.ArgumentParser(description="同步知识库索引（向量 + BM25）")
    parser.add_argument("--rebuild", action="store_true", help="清空索引后全量重建")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个待同步文件")
    parser.add_argument("--status", action="store_true", help="只打印索引状态")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    setup_logging()
    resolved = settings or get_settings()
    kb = KnowledgeBase(resolved)
    try:
        if args.status:
            print(json.dumps(kb.status(), ensure_ascii=False, indent=2))
            return 0

        report = kb.sync(rebuild=args.rebuild, limit=args.limit)
        if args.json:
            print(json.dumps({"report": report.to_dict(), "status": kb.status()}, ensure_ascii=False, indent=2))
        else:
            print(f"同步结果：{report.summary()}")
            status = kb.status()
            print(
                "索引状态："
                f"文档 {status['documents']} / 向量 {status['vector_chunks']} / "
                f"BM25 {status['bm25_chunks']} / collection={status['collection']} / "
                f"embedder={status['embedder']}{'（mock）' if status['embedder_mock'] else ''}"
            )
            for failure in report.failed:
                print(f"  [失败] {failure['file']}（{failure['stage']}）：{failure['error']}")
        return 0 if report.ok else 1
    finally:
        kb.close()


if __name__ == "__main__":
    raise SystemExit(main())
