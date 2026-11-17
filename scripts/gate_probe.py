r"""探测拒答闸门的实际工作点：每题的分数、词面覆盖、被哪道闸门拦下、放宽线该取多少。

为什么需要它：`scripts/ablation.py` 告诉你**误拒率是多少**，但不告诉你**为什么**。
73 题实测遇到过一次——把 `coverage_relax_score` 从 0.75 改成默认值之后重跑消融表，
一位数字都没变。原因只有把每题的 `best_score` 与 `lexical_coverage` 摊开才看得见：
真实 bge-reranker-v2-m3 在这批题上最高只给到 0.727，0.75 那条线从来没被跨过。
**没跑过分布就写下的阈值，等于没写。**

这条链路只跑**检索**（不生成、不调 LLM），所以：

- 快：73 题不到一分钟（模型加载另算），可以反复调参；
- 口径是检索层：这里的"误拒"是 `has_evidence=False`，端到端（含逐句校验与生成侧改判）
  的数字看 `scripts/ablation.py`；
- 扫描表是**离线算**出来的：拿记录下来的 `best_score` / `lexical_coverage` 按不同放宽线
  重放判定，不需要按每条线重跑一次检索。

用法::

    # 48 篇评测库：语料目录和向量库都要指过去，否则查的是 data/raw 的演示库
    $env:RAG_SYNC__WATCH_DIR="data/bench/raw"; $env:RAG_VECTORSTORE__PERSIST_DIR="data/bench/index/chroma"
    python scripts/gate_probe.py --label bench --out reports/gate-probe-bench.md `
      --golden data/bench/golden_A.jsonl --golden data/bench/golden_B.jsonl

    python scripts/gate_probe.py --golden data/eval/golden_set.jsonl --lines 0.5,0.55,0.6,0.65 --json

退出码：0 正常；2 索引为空（先跑 `scripts/sync.py --rebuild` 或 `scripts/bench.py --rebuild`）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import evaluate as ev  # noqa: E402  复用题集加载，别让两套 golden 解析各写一遍
from rag_assistant.chat_service import ChatService  # noqa: E402
from rag_assistant.config import get_settings  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402

#: 放宽线默认扫描点。0 有特殊含义（永不放宽 = 旧行为），必须包含，否则
#: "改了默认值到底有没有用"这件事就只存在于口头承诺里。
DEFAULT_LINES = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


@dataclass(frozen=True)
class Probe:
    """一道题在检索层的判定现场。"""

    question: str
    variant: str
    domain: str
    should_refuse: bool
    best_score: float
    gate: str
    coverage: float
    content_tokens: int
    has_evidence: bool
    reason: str
    top_doc: str
    #: 本次跑的精排是不是 mock。放宽的前提要一题一题记下来，
    #: 否则扫描表会承诺一个线上拿不到的行为（mock 分数与 sigmoid 精排分不同尺度）。
    reranker_mock: bool = False

    @property
    def group(self) -> str:
        if self.should_refuse:
            return "应拒答"
        return f"可答/{self.variant or 'unlabeled'}"

    @property
    def relax_available(self) -> bool:
        """这一题是否真的拿到了可信精排分——检索层只在 `gate == "rerank"` 时才考虑放宽。"""
        return self.gate == "rerank" and not self.reranker_mock


def percentile(values: Sequence[float], position: float) -> float:
    """最近秩百分位（不插值）。样本只有几道题时，插值出来的小数点全是假的精度。

    取整用"加 0.5 向下取整"而不是 `round()`：Python 的 round 是银行家舍入，
    样本数为 3 时 p25 会落到 min，报告里同一格子的含义会随题数抖一下。
    """
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.floor(position * (len(ordered) - 1) + 0.5)))
    return ordered[index]


def describe(values: Sequence[float]) -> str:
    if not values:
        return "-"
    return (
        f"n={len(values)} min={min(values):.3f} p25={percentile(values, 0.25):.3f} "
        f"中位={statistics.median(values):.3f} p75={percentile(values, 0.75):.3f} max={max(values):.3f}"
    )


def probe_all(paths: Sequence[Path]) -> tuple[list[Probe], dict[str, Any]]:
    """对每题跑一次检索，把闸门要用的数都记下来；第二项是本次跑的实际环境。

    环境必须从**活着的 service** 读，不能另开一份配置：配置里写 0.55、实际加载的是
    mock 精排或另一个索引，扫描表就成了小说。
    """
    settings = get_settings()
    items: list[ev.GoldenItem] = []
    for path in paths:
        items.extend(ev.load_golden(path))
    if not items:
        raise SystemExit("题集是空的")

    service = ChatService(settings)
    probes: list[Probe] = []
    try:
        status = service.kb.status()
        if status["vector_chunks"] == 0:
            print(f"索引为空（{status['raw_dir']} → {settings.chroma_dir}）：\n"
                  "  先跑 python scripts\\sync.py --rebuild（演示库）或 python scripts/bench.py（48 篇评测库）",
                  file=sys.stderr)
            raise SystemExit(2)
        reranker = service.kb.reranker
        for item in items:
            result = service.kb.search(item.question)
            diag = result.diagnostics
            probes.append(
                Probe(
                    question=item.question,
                    variant=item.variant,
                    domain=item.domain,
                    should_refuse=item.should_refuse,
                    best_score=result.best_score,
                    gate=str(diag.get("gate") or "none"),
                    coverage=float(diag.get("lexical_coverage") or 0.0),
                    content_tokens=len(diag.get("content_tokens") or []),
                    has_evidence=result.has_evidence,
                    reason=str(diag.get("reason") or ""),
                    top_doc=result.chunks[0].doc_id if result.chunks else "-",
                    reranker_mock=bool(reranker is not None and reranker.is_mock),
                )
            )
    finally:
        service.close()
    env = {
        "Embedding": f"{service.kb.embedder.name}（mock={service.kb.embedder.is_mock}）",
        "Reranker": f"{reranker.name if reranker else '关闭'}（mock={getattr(reranker, 'is_mock', None)}）",
        "索引": f"{status['collection']}｜{status['vector_chunks']} chunk｜{status['documents']} 文档",
        "语料目录": str(status["raw_dir"]),
        "score_threshold": settings.retrieval.score_threshold,
        "min_lexical_coverage": settings.retrieval.min_lexical_coverage,
        "coverage_relax_score": settings.retrieval.coverage_relax_score,
    }
    if env["Embedding"].endswith("mock=True）") or env["Reranker"].endswith("mock=True）"):
        print("警告：本次是 mock 模型，分数只能验证脚本跑得通，不能用来标定阈值。", file=sys.stderr)
    return probes, env


def simulate(probes: Sequence[Probe], threshold: float, min_coverage: float, relax_line: float) -> dict[str, int]:
    """按给定放宽线重放判定：`relax_line=0` 表示永不放宽（旧行为）。

    三个参数都从配置取，重放用的判定顺序与 `retrieval/retriever.py::_retrieve` 一致：
    先看分数，再看词面覆盖（覆盖够不够只在**未被放宽**时才起作用）。
    """
    false_reject = correct_refuse = answered = leaked = 0
    for probe in probes:
        relaxed = bool(relax_line > 0 and probe.relax_available and probe.best_score >= relax_line)
        off_topic = bool(min_coverage > 0 and not relaxed and probe.content_tokens > 0
                         and probe.coverage < min_coverage)
        refused = probe.best_score < threshold or off_topic
        if probe.should_refuse:
            correct_refuse += int(refused)
            leaked += int(not refused)
        else:
            false_reject += int(refused)
            answered += int(not refused)
    return {
        "relax_line": relax_line,
        "false_reject": false_reject,
        "answered": answered,
        "correct_refuse": correct_refuse,
        "leaked": leaked,
    }


def render(probes: Sequence[Probe], meta: dict[str, Any]) -> str:
    answerable = [p for p in probes if not p.should_refuse]
    refuse = [p for p in probes if p.should_refuse]
    lines = [
        f"# 拒答闸门现场 · {meta['label']}",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 0. 运行条件",
        "",
    ]
    for key, value in meta.items():
        if key != "label":
            lines.append(f"- {key}：{value}")
    lines += [
        "",
        "**只跑检索、不跑生成**：这里的误拒/正确拒答是闸门 1（阈值）与闸门 2（词面覆盖）的口径，",
        "端到端数字（含逐句校验改判）看 `scripts/ablation.py` 的报告。",
        "",
        "## 1. 分数分布",
        "",
        "| 分组 | 最优分分布 | 词面覆盖 |",
        "|---|---|---|",
    ]
    for group in ("可答/literal", "可答/paraphrase", "可答/unlabeled", "应拒答"):
        subset = [p for p in probes if p.group == group]
        if not subset:
            continue
        lines.append(
            f"| {group} | {describe([p.best_score for p in subset])} "
            f"| {describe([p.coverage for p in subset])} |"
        )
    lines += [
        "",
        "两个分布**重叠**就说明单点阈值切不开，别再调小数点；能把可答题与应拒题分开的",
        "只有更好的精排或语义级判定。词面覆盖一列是闸门 2 的工作现场：可答题的覆盖也压得很低时，",
        "闸门 2 就是在误拒真人提问，而不是在挡无关问题。",
        "",
        "## 2. 放宽线扫描（0 = 永不放宽）",
        "",
        f"阈值固定 {meta['score_threshold']}，词面覆盖下限固定 {meta['min_lexical_coverage']}。",
        "",
        "| 放宽线 | 误拒题数 | 答上题数 | 正确拒答 | 漏拒（该拒却答了） |",
        "|---|---|---|---|---|",
    ]
    for line in meta["lines"]:
        sim = simulate(probes, float(meta["score_threshold"]), float(meta["min_lexical_coverage"]), float(line))
        mark = " ←" if float(line) == float(meta["coverage_relax_score"]) else ""
        lines.append(
            f"| {line}{mark} | {sim['false_reject']}/{len(answerable)} | {sim['answered']} "
            f"| {sim['correct_refuse']}/{len(refuse)} | {sim['leaked']} |"
        )
    lines += [
        "",
        "读法：往左移一格放宽线，通常同时**减少误拒**和**增加漏拒**，"
        "两个数各自的增量就是这道闸门的边际价格；把它们跟当前默认值对齐着看。",
        "",
        "## 3. 明细",
        "",
        "| 问题 | 分组 | 领域 | 最优分 | 覆盖 | 内容词 | gate | 判定 | 首位文档 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for probe in probes:
        question = probe.question if len(probe.question) <= 28 else probe.question[:27] + "…"
        lines.append(
            f"| {question} | {probe.group} | {probe.domain or '-'} | {probe.best_score:.4f} "
            f"| {probe.coverage:.2f} | {probe.content_tokens} | {probe.gate} | {probe.reason} | {probe.top_doc} |"
        )
    lines += [
        "",
        "## 4. 现在被 off_topic 拦下的可答题",
        "",
    ]
    stuck = [p for p in answerable if p.reason == "off_topic"]
    if not stuck:
        lines.append("- 没有：词面闸门当前没有误拒可答题。")
    else:
        lines.append(
            f"- {len(stuck)} 题，最优分区间 {min(p.best_score for p in stuck):.3f}–"
            f"{max(p.best_score for p in stuck):.3f}，覆盖区间 "
            f"{min(p.coverage for p in stuck):.2f}–{max(p.coverage for p in stuck):.2f}。"
        )
        lines.append("- 这些题的期望文档**多数已经在候选第一位**（看 §3 的首位文档），")
        lines.append("  也就是说材料送到了、闸门 2 把它拒了——不是检索没命中。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索层拒答闸门现场与放宽线扫描")
    parser.add_argument(
        "--golden",
        action="append",
        default=[],
        help="题集文件，可重复（默认 data/eval/golden_set.jsonl）",
    )
    parser.add_argument("--label", default="gate", help="报告标签")
    parser.add_argument("--lines", default=",".join(str(x) for x in DEFAULT_LINES), help="放宽线扫描点，逗号分隔")
    parser.add_argument("--out", type=Path, default=None, help="markdown 输出路径")
    parser.add_argument("--json", action="store_true", help="把明细与扫描打到 stdout")
    args = parser.parse_args(argv)

    setup_logging()
    paths = [Path(p) for p in args.golden] or [PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"]
    paths = [p if p.is_absolute() else PROJECT_ROOT / p for p in paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"找不到题集：{missing}", file=sys.stderr)
        return 2
    lines = [float(x) for x in args.lines.split(",") if x.strip()]
    if not lines:
        print("--lines 解析为空", file=sys.stderr)
        return 2

    probes, env = probe_all(paths)

    answerable = [p for p in probes if not p.should_refuse]
    refuse = [p for p in probes if p.should_refuse]
    meta = {
        "label": args.label,
        "题集": "、".join(p.name for p in paths),
        "题数": len(probes),
        "可答": len(answerable),
        "应拒答": len(refuse),
        **env,
        "lines": lines,
    }
    report = render(probes, meta)
    if args.out:
        out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"报告已写入：{out_path}")
    else:
        print(report)
    if args.json:
        payload = {
            "meta": meta,
            "sweep": [
                simulate(probes, float(meta["score_threshold"]), float(meta["min_lexical_coverage"]), line)
                for line in lines
            ],
            "probes": [p.__dict__ for p in probes],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
