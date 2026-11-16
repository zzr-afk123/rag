"""检索调试 CLI（阶段 B 的验收工具）。

用法::

    python scripts/search.py "休学最长可以多久"
    python scripts/search.py "在读证明怎么开" --top-n 3 --json
    python scripts/search.py "报销标准" --extra "差旅费限额" --extra "会议费标准"

输出每一跳的分数（vector / bm25 / rrf / rerank），方便判断"是召回问题还是排序问题"。
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

from rag_assistant.kb import KnowledgeBase  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="混合检索调试")
    parser.add_argument("query", help="查询语句")
    parser.add_argument("--top-n", type=int, default=None, help="返回条数")
    parser.add_argument("--extra", action="append", default=[], help="额外等价查询（可多次）")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--full-text", action="store_true", help="打印完整正文")
    parser.add_argument("--threshold", type=float, default=None, help="临时覆盖相关性阈值（标定用）")
    args = parser.parse_args(argv)

    setup_logging()
    kb = KnowledgeBase()
    try:
        result = kb.search(
            args.query, top_n=args.top_n, extra_queries=args.extra, threshold=args.threshold
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "query": result.query,
                        "has_evidence": result.has_evidence,
                        "best_score": result.best_score,
                        "diagnostics": result.diagnostics,
                        "chunks": [chunk.to_dict() for chunk in result.chunks],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        diagnostics = result.diagnostics
        print(
            f"\n查询：{result.query}\n"
            f"候选：向量 {diagnostics.get('vector_hits')} / BM25 {diagnostics.get('bm25_hits')} "
            f"→ 融合 {diagnostics.get('fused')} → 重排 {diagnostics.get('reranked')}"
        )
        print(
            f"判定：{'有依据' if result.has_evidence else '证据不足（应拒答）'}  "
            f"最优分 {result.best_score:.4f} / 门槛 {diagnostics.get('threshold')}  "
            f"口径 {diagnostics.get('gate')}  耗时 {diagnostics.get('latency_ms')}ms"
        )
        if diagnostics.get("note"):
            print(f"提示：{diagnostics['note']}")
        for index, chunk in enumerate(result.chunks, start=1):
            scores = [
                f"vec={chunk.vector_score:.3f}" if chunk.vector_score is not None else "vec=-",
                f"bm25={chunk.bm25_score:.2f}" if chunk.bm25_score is not None else "bm25=-",
                f"rrf={chunk.rrf_score:.4f}",
                f"rerank={chunk.rerank_score:.4f}" if chunk.rerank_score is not None else "rerank=-",
            ]
            print(f"\n[{index}] {chunk.doc_name}  <-  {chunk.location}")
            print(f"    {'  '.join(scores)}  chunk={chunk.chunk_id}")
            body = chunk.text if args.full_text else chunk.text.replace("\n", " ")[:120]
            print(f"    {body}")
        return 0
    finally:
        kb.close()


if __name__ == "__main__":
    raise SystemExit(main())
