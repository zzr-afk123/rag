r"""消融表：一次把 5 行配置跑完，输出能直接贴进 README 的 markdown 表格。

行是照着"每一行只多开一个东西"设计的，出问题时能一眼看出是哪一层在掉链子：

| 行 | 配置 | 想说明什么 |
|---|---|---|
| BM25 only | 向量权重 0 | 纯词面召回的地板 |
| 向量 only | BM25 权重 0 | 纯语义召回的地板 |
| +RRF | 两路融合、关掉 reranker | 融合本身值多少分 |
| +rerank | 默认完整链路 | 重排值多少分 |
| 关闸门 | 阈值 0 + 词面覆盖 0 + 逐句校验 off | 闸门关掉之后会答出多少不该答的 |

`weight_*=0` 是真消融：``rrf_fuse`` 对 ``weight <= 0`` 的排序直接跳过（见
``src/rag_assistant/retrieval/fusion.py``），所以"向量 only"确实一个字面通道都不走。

用法::

    # 先把 bench 语料索引到独立目录，免得动演示库
    RAG_SYNC__WATCH_DIR=data/bench/raw RAG_VECTORSTORE__PERSIST_DIR=data/bench/index/chroma \
        python scripts/sync.py --rebuild
    python scripts/ablation.py --golden data/bench/golden_A.jsonl --golden data/bench/golden_B.jsonl

只想快速看链路：加 ``RAG_FORCE_MOCK=1``（embedder/reranker 都退化成 mock，
数字只能证明脚本跑得通，**不能拿去比**）。环境表会把是否 mock 打印出来。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import evaluate as ev  # noqa: E402
from rag_assistant.chat_service import ChatService  # noqa: E402
from rag_assistant.config import Settings, get_settings  # noqa: E402


@dataclass(frozen=True)
class Row:
    name: str
    purpose: str
    overrides: tuple[str, ...] = ()
    #: 显式传给回答环节的阈值；None = 用配置里的 score_threshold
    threshold: float | None = None


ROWS: tuple[Row, ...] = (
    Row("BM25 only", "纯词面召回的地板", ("retrieval.weight_vector=0",)),
    Row("向量 only", "纯语义召回的地板", ("retrieval.weight_bm25=0",)),
    Row("+RRF", "两路融合、不重排", ("reranker.enabled=false",)),
    Row("+rerank", "完整链路（默认配置）"),
    Row(
        "关闸门",
        "阈值、词面覆盖、逐句校验全关：看闸门拦住了什么",
        ("retrieval.min_lexical_coverage=0", "generation.claim_check=off"),
        threshold=0.0,
    ),
)


def run_row(
    base: Settings,
    row: Row,
    items: Sequence[ev.GoldenItem],
    *,
    judge_mode: str,
) -> dict[str, Any]:
    """跑一行，返回指标 + 实际生效的配置（配置没生效的话这行就是假数据）。

    每行新建一整套 service/kb，不复用：``ChatService.close()`` 会连带关掉 kb
    里的向量库连接，共享实例会让第一行把后面四行的地基关掉。代价是模型重复加载
    （bge-m3 没有进程内缓存），换来的是"这行的配置真的只作用于这行"。
    """
    settings, applied = ev.apply_overrides(base, row.overrides)
    service = ChatService(settings)
    try:
        results = [
            ev.evaluate_item(service, item, judge_mode=judge_mode, threshold=row.threshold) for item in items
        ]
    finally:
        service.close()
    metrics = ev.compute_metrics(results)
    return {
        "name": row.name,
        "purpose": row.purpose,
        "applied": applied,
        "effective": {
            "weight_vector": settings.retrieval.weight_vector,
            "weight_bm25": settings.retrieval.weight_bm25,
            "reranker": settings.reranker.enabled,
            "min_lexical_coverage": settings.retrieval.min_lexical_coverage,
            "score_threshold": row.threshold if row.threshold is not None else settings.retrieval.score_threshold,
            "claim_check": settings.generation.claim_check,
        },
        "metrics": metrics,
    }


def _pct(value: float | None, digits: int = 1, *, percent: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%" if percent else f"{value:.{digits}f}"


def _ratio(value: float | None, denominator: int) -> str:
    """分母为 0 时不能写成 0.0%——那看起来像"全错"，实际是"没这类题"。"""
    return "-" if not denominator else _pct(value)


def _cell(row: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {row} | {_pct(metrics['hit@1'])} | {_pct(metrics['hit@3'])} "
        f"| {_pct(metrics['answer_hit@1'])} | {metrics['answer_mrr']:.3f} "
        f"| {_ratio(metrics['point_coverage'], metrics['n_answerable'])} "
        f"| {_ratio(metrics['false_reject_rate'], metrics['n_answerable'])} "
        f"| {_ratio(metrics['refusal_recall'], metrics['n_refuse'])} "
        f"| {metrics['latency_p50_ms']:.0f}ms |"
    )


_HEADER = "| 配置 | 候选 Hit@1 | 候选 Hit@3 | 答案 Hit@1 | 答案 MRR | 要点覆盖 | 误拒率 | 正确拒答率 | P50 |"
_RULE = "|---|---|---|---|---|---|---|---|---|"


def render(rows: list[dict[str, Any]], items: Sequence[ev.GoldenItem], meta: dict[str, Any]) -> str:
    lines = [
        f"# 消融表 · {meta['label']}",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 0. 运行条件",
        "",
    ]
    for key, value in meta.items():
        lines.append(f"- {key}：{value}")
    lines += [
        "",
        "「答案 Hit@1」是在**回答正文的引用序列**上判的：期望文档必须真的出现在答案里，",
        "光被召回不算。语料少于几十篇时两个口径会几乎相等，那说明的是题目太简单，不是效果好。",
        "",
        "## 1. 总体",
        "",
        _HEADER,
        _RULE,
    ]
    for row in rows:
        lines.append(_cell(row["name"], row["metrics"]))
    lines += ["", "## 2. 每行实际生效的配置", "", "| 行 | 覆盖参数 | weight_vector | weight_bm25 | reranker | 词面覆盖 | 阈值 | claim_check |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        eff = row["effective"]
        lines.append(
            f"| {row['name']} | {'、'.join(row['applied']) or '（默认）'} "
            f"| {eff['weight_vector']} | {eff['weight_bm25']} | {eff['reranker']} "
            f"| {eff['min_lexical_coverage']} | {eff['score_threshold']} | {eff['claim_check']} |"
        )

    section = 2
    for title, attr, metric_key in (("改写强度", "variant", "by_variant"), ("领域", "domain", "by_domain")):
        if not any(getattr(item, attr) for item in items):
            continue
        section += 1
        lines += ["", f"## {section}. 按{title}拆开", ""]
        for row in rows:
            lines += [
                f"### {row['name']}",
                "",
                "| 分组 | 题数 | 候选 Hit@1 | 答案 Hit@1 | 答案 MRR | 要点覆盖 | 误拒率 | 正确拒答率 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for group in row["metrics"].get(metric_key) or []:
                lines.append(
                    f"| {group['name']} | {group['n']} | {_pct(group['hit@1'])} "
                    f"| {_pct(group['answer_hit@1'])} | {_pct(group['answer_mrr'], 3, percent=False)} "
                    f"| {_pct(group['point_coverage'])} | {_pct(group['false_reject_rate'])} "
                    f"| {_pct(group['refusal_recall'])} |"
                )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索/闸门消融表")
    parser.add_argument(
        "--golden",
        action="append",
        default=[],
        help="题集文件，可重复（默认 data/eval/golden_set.jsonl）",
    )
    parser.add_argument("--label", default="bench", help="报告标签")
    parser.add_argument("--limit", type=int, default=None, help="每个配置只跑前 N 题")
    parser.add_argument("--judge", choices=["rule", "llm", "none"], default="rule")
    parser.add_argument("--only", action="append", default=[], help="只跑指定行（按名字，可重复）")
    parser.add_argument("--out", type=Path, default=None, help="markdown 输出路径")
    parser.add_argument("--json", action="store_true", help="把每行指标打到 stdout（供脚本消费）")
    args = parser.parse_args(argv)

    ev.setup_logging()
    base = get_settings()
    paths = [Path(p) for p in args.golden] or [PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"]
    items: list[ev.GoldenItem] = []
    for path in paths:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        items.extend(ev.load_golden(path))
    if args.limit:
        items = items[: args.limit]

    rows_wanted = [row for row in ROWS if not args.only or row.name in args.only]
    if not rows_wanted:
        print(f"--only 没匹配到任何行，可选：{[row.name for row in ROWS]}", file=sys.stderr)
        return 2

    probe = ChatService(base)
    status = probe.kb.status()
    if status["vector_chunks"] == 0:
        probe.close()
        print(
            f"索引为空（{status['raw_dir']} → {base.chroma_dir}）："
            "先跑 python scripts/sync.py --rebuild",
            file=sys.stderr,
        )
        return 2
    if status["embedder_mock"]:
        print(
            "警告：embedder 处于 mock 模式，本次消融只能验证脚本，数字不可用于对比。"
            "真实跑请先安装 requirements-model.txt。",
            file=sys.stderr,
        )
    probe.close()

    rows: list[dict[str, Any]] = []
    for row in rows_wanted:
        print(f"[ablation] {row.name} · {len(items)} 题 · {', '.join(row.overrides) or '默认'}")
        result = run_row(base, row, items, judge_mode=args.judge)
        metrics = result["metrics"]
        print(
            f"           候选 Hit@1 {_pct(metrics['hit@1'])}｜答案 Hit@1 {_pct(metrics['answer_hit@1'])}｜"
            f"误拒 {_pct(metrics['false_reject_rate'])}｜拒答召回 {_pct(metrics['refusal_recall'])}"
        )
        rows.append(result)

    meta = {
        "label": args.label,
        "题集": "、".join(str(Path(p).name) for p in paths),
        "题数": len(items),
        "语料目录": str(base.raw_dir.relative_to(PROJECT_ROOT)),
        "向量库": f"{status['collection']}（{status['vector_chunks']} chunk / {status['documents']} 文档）",
        "Embedding": f"{base.embedding.model}",
        "Reranker": f"{base.reranker.model}（enabled={base.reranker.enabled}）",
        "忠实度判定": args.judge,
    }
    report = render(rows, items, meta)
    out_path = args.out or (
        PROJECT_ROOT / "reports" / f"ablation-{args.label}-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{out_path}")
    if args.json:
        print(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
