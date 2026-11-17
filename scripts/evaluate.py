"""阶段 F：评测脚本。

产出（默认写到 ``reports/eval-<label>-<时间>.md``）：

- **检索质量**：Hit@1 / Hit@3 / Hit@5、MRR（以 ``expected_doc`` 是否命中为准）
- **拒答质量**：混淆矩阵、正确拒答率、误拒率、漏答率、总体准确率
- **回答质量**：要点覆盖率（expected_answer_points 能否在**被引用的原文**里找到）
  与忠实度（规则判定或 LLM-as-judge）
- **阈值标定**：扫描 ``score_threshold``，给出"误拒 vs 漏答"的权衡曲线与推荐值

用法::

    python scripts/evaluate.py                       # 用当前配置（真实模型或 mock）
    python scripts/evaluate.py --label bge-m3-real
    python scripts/evaluate.py --threshold 0.5 --judge llm
    python scripts/evaluate.py --limit 3             # 只跑前 3 条（调试）

注意：
1. 评测跑的是**检索 + 生成**全链路，但每个问题独立会话（不带历史），
   避免上一题污染下一题；
2. 生成阶段用 ``threshold=0`` 跑一遍（保证进入回答环节），
   阈值扫描是基于记录下来的 best_score 离线计算的——**只覆盖闸门 1**，
   闸门 2（引用校验）不受阈值影响；
3. ``--judge llm`` 需要非 mock 的 LLM（DeepSeek Key）；mock LLM 会自动回落到规则判定。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_assistant.chat_service import ChatService  # noqa: E402
from rag_assistant.config import get_settings  # noqa: E402
from rag_assistant.generation.answerer import AnswerResult  # noqa: E402
from rag_assistant.llm.base import LLMClient  # noqa: E402
from rag_assistant.llm.prompts import judge_prompt  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402

HIT_KS = (1, 3, 5)
#: 报告里最多展示多少条无依据句子（避免诊断刷屏）
_MAX_REPORT = 20
SWEEP_THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 19)]  # 0.05 ~ 0.90


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
@dataclass
class GoldenItem:
    question: str
    expected_doc: str = ""
    expected_sections: list[str] = field(default_factory=list)
    expected_answer_points: list[str] = field(default_factory=list)
    should_refuse: bool = False
    note: str = ""
    #: 允许多个"都算对"的文档（实测发现同一事实可能分散在 docx 正文与 xlsx 表格里，
    #: 只认一个文档会把"更直接的回答"判成检索失败）。
    expected_docs: list[str] = field(default_factory=list)
    #: 题目改写强度：`literal` = 只换同义词/调语序（检索压力小）；
    #: `paraphrase` = 换人称换措辞、像真人提问（更接近线上）。两套题**分开报**，
    #: 才能看出短板到底在检索还是在生成；留空 = 未标注，报告会单列成 unlabeled，
    #: 不会把未标注的题悄悄混进上面两类。
    variant: str = ""
    #: 领域/主题分组，消融表按它拆开看
    domain: str = ""

    @property
    def acceptable_docs(self) -> set[str]:
        docs = {doc for doc in self.expected_docs if doc}
        if self.expected_doc:
            docs.add(self.expected_doc)
        return docs


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number} JSON 解析失败：{exc}") from exc
        if not payload.get("question"):
            raise SystemExit(f"{path}:{number} 缺少 question")
        items.append(
            GoldenItem(
                question=payload["question"],
                expected_doc=payload.get("expected_doc", ""),
                expected_docs=list(payload.get("expected_docs") or []),
                expected_sections=list(payload.get("expected_sections") or []),
                expected_answer_points=list(payload.get("expected_answer_points") or []),
                should_refuse=bool(payload.get("should_refuse", False)),
                note=payload.get("note", ""),
                variant=str(payload.get("variant") or "").strip(),
                domain=str(payload.get("domain") or "").strip(),
            )
        )
    return items


# --------------------------------------------------------------------------
# 单题结果
# --------------------------------------------------------------------------
@dataclass
class ItemResult:
    item: GoldenItem
    answer: str
    refused: bool
    refusal_reason: str | None
    best_score: float
    rank: int | None  # expected_doc 在候选中的名次（1-based）
    #: expected_doc 在**答案实际引用的资料**里的序号（1-based），即正文里第几个 [n]
    #: 才是对的那篇；None = 答案没引用到期望文档。这就是"在答案正文上判的 Hit@1"。
    answer_rank: int | None
    citations: list[dict[str, Any]]
    cited_texts: list[str]
    point_hits: dict[str, bool]
    faithful: bool | None
    judge_reason: str
    judge: str
    latency_ms: float
    usage_prompt: int
    usage_completion: int
    # ---- claim 级引用校验（闸门 2.5）----
    claim_support_rate: float | None = None
    claim_mode: str = ""
    claim_corrections: list[dict[str, Any]] = field(default_factory=list)
    claim_dropped: list[dict[str, Any]] = field(default_factory=list)
    claim_unsupported: list[dict[str, Any]] = field(default_factory=list)

    @property
    def hit_at(self) -> dict[int, bool]:
        return {k: bool(self.rank and self.rank <= k) for k in HIT_KS}

    @property
    def answer_hit_at(self) -> dict[int, bool]:
        return {k: bool(self.answer_rank and self.answer_rank <= k) for k in HIT_KS}

    @property
    def refusal_correct(self) -> bool:
        return self.refused == self.item.should_refuse


def _rank_of_expected(record: AnswerResult, acceptable: set[str]) -> int | None:
    """期望文档在候选中的名次（1-based）；``acceptable`` 里任一命中即算。"""
    if not acceptable:
        return None
    for index, chunk in enumerate(record.retrieved, start=1):
        if str(chunk.get("doc_name", "")) in acceptable:
            return index
    return None


def _answer_rank_of_expected(record: AnswerResult, acceptable: set[str]) -> int | None:
    """期望文档在**答案正文**中的引用序号。

    候选名次衡量"检索有没有捞到"，但捞到之后可能被上下文的 token 预算挤掉，
    或者模型压根没引它——这两件事在只有 6~11 篇语料时看不出来（候选里全是相关
    文档，Hit@1 自然虚高）。按引用序号判就把闸门、预算裁剪、模型的引用选择全部
    算进成本，数字会难看，但那个才是能拿去跟别人比的。
    """
    if not acceptable:
        return None
    for position, citation in enumerate(record.citations, start=1):
        if str(citation.doc_name or "") in acceptable:
            return int(citation.index) if citation.index > 0 else position
    return None


def _cited_contexts(record: AnswerResult) -> tuple[list[dict[str, Any]], list[str]]:
    """把引用编号映射回**模型实际看到的资料块正文**。

    注意 1：上下文已按"文档+章节"合并（一个章节 = 一个资料块 = 一个引用编号），
    所以必须按章节去 ``context_blocks`` 里取 body，而不是按 chunk_id 取单条
    chunk 的文本——否则会把"表格+正文"的综合回答误判成无依据（实测第 6/27 题
    就栽在这里）。

    注意 2（曾经的真 bug）：``Citation.index`` 是**重编号后**的引用序号
    （按首次引用顺序 1..N），**不等于** prompt 里的资料块编号。早先版本按
    ``citation.index`` 去查 ``context_blocks``，当答案只引第 2 块时就会取到
    第 1 块的正文，LLM 评审判"原文里找不到"，要点覆盖也归零——第 9、12 题
    的"不忠实"其实是这样来的。现在按 ``(doc_id, section_path)`` 匹配，
    其次 chunk_id，最后才退回重编号与摘录。
    """
    by_key = {
        (str(block.get("doc_id", "")), str(block.get("section_path", ""))): block
        for block in record.context_blocks
    }
    by_id = {str(chunk.get("chunk_id")): chunk for chunk in record.retrieved}
    citations: list[dict[str, Any]] = []
    texts: list[str] = []
    for citation in record.citations:
        payload = citation.to_dict()
        block = by_key.get((citation.doc_id, citation.section_path))
        if block:
            text = str(block.get("body", ""))
        else:
            source = by_id.get(citation.chunk_id, {})
            text = str(source.get("text", "")) or citation.quote
        payload["source_text"] = text
        citations.append(payload)
        texts.append(text)
    return citations, texts


def _point_coverage(points: Iterable[str], haystack: str) -> dict[str, bool]:
    compact = "".join(haystack.split())
    return {point: point in compact or "".join(point.split()) in compact for point in points}


def _judge_by_rule(points: dict[str, bool], has_citation: bool) -> tuple[bool, str]:
    if not points:
        return (has_citation, "无要点可校验" if has_citation else "没有任何引用")
    missing = [point for point, found in points.items() if not found]
    if missing:
        return False, "以下要点未在引用原文中找到：" + "、".join(missing)
    return True, "全部要点均可在引用原文中找到"


def _judge_by_llm(
    llm: LLMClient, item: GoldenItem, answer: str, cited_texts: Sequence[str]
) -> tuple[bool | None, str]:
    """LLM-as-judge；模型是 mock 或输出不可解析时返回 None（交给规则判定）。"""
    if llm.is_mock:
        return None, "mock LLM 不做语义评审"
    try:
        response = llm.complete(judge_prompt(item.question, answer, cited_texts), temperature=0.0, max_tokens=400)
    except Exception as exc:  # noqa: BLE001 - 评审失败不应让评测中断
        return None, f"评审调用失败：{type(exc).__name__}"

    text = response.text or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, f"评审输出非 JSON：{text[:60]}"
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None, f"评审 JSON 解析失败：{text[:60]}"
    return bool(payload.get("faithful", False)), str(payload.get("reason", ""))


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def _hit_metrics(results: Sequence[ItemResult], attr: str) -> dict[str, Any]:
    """按某个名次字段（``rank`` 或 ``answer_rank``）算 Hit@k 与 MRR。"""
    answerable = [r for r in results if not r.item.should_refuse]
    hits = {
        k: (sum(1 for r in answerable if getattr(r, attr) and getattr(r, attr) <= k) / len(answerable))
        if answerable
        else 0.0
        for k in HIT_KS
    }
    reciprocal = [1.0 / value for value in (getattr(r, attr) for r in answerable) if value]
    return {
        **{f"hit@{k}": hits[k] for k in HIT_KS},
        "mrr": statistics.fmean(reciprocal) if reciprocal else 0.0,
    }


def compute_metrics(results: Sequence[ItemResult]) -> dict[str, Any]:
    answerable = [r for r in results if not r.item.should_refuse]
    refuse_items = [r for r in results if r.item.should_refuse]

    retrieval = _hit_metrics(results, "rank")
    cited = _hit_metrics(results, "answer_rank")

    tp = sum(1 for r in refuse_items if r.refused)
    fn = sum(1 for r in refuse_items if not r.refused)
    fp = sum(1 for r in answerable if r.refused)
    tn = sum(1 for r in answerable if not r.refused)
    total = max(1, len(results))

    judged = [r for r in results if r.faithful is not None]
    answer_points = [point for r in answerable for point in r.point_hits]
    covered = sum(1 for r in answerable for found in r.point_hits.values() if found)

    latencies = sorted(r.latency_ms for r in results)
    # claim 级校验只统计真正作答的题（拒答题没有句子可核）
    claim_checked = [r for r in results if r.claim_support_rate is not None and not r.refused]
    claim_mean = (
        statistics.fmean(r.claim_support_rate for r in claim_checked) if claim_checked else None
    )
    return {
        "n": len(results),
        "n_answerable": len(answerable),
        "n_refuse": len(refuse_items),
        "hit@1": retrieval["hit@1"],
        "hit@3": retrieval["hit@3"],
        "hit@5": retrieval["hit@5"],
        "mrr": retrieval["mrr"],
        # 答案正文口径：期望文档落在答案的第几个引用上（没引到 = 不算命中）
        "answer_hit@1": cited["hit@1"],
        "answer_hit@3": cited["hit@3"],
        "answer_mrr": cited["mrr"],
        "refusal_tp": tp,
        "refusal_fn": fn,
        "refusal_fp": fp,
        "refusal_tn": tn,
        "refusal_accuracy": (tp + tn) / total,
        "refusal_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "refusal_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "false_reject_rate": fp / len(answerable) if answerable else 0.0,
        "point_coverage": (covered / len(answer_points)) if answer_points else 0.0,
        "faithfulness": (
            sum(1 for r in judged if r.faithful) / len(judged) if judged else None
        ),
        # claim 级校验：只看真正作答的题（拒答题没有句子可核）
        "claim_support_rate": claim_mean,
        "claim_corrections": sum(len(r.claim_corrections) for r in results),
        "claim_dropped": sum(len(r.claim_dropped) for r in results),
        "claim_unsupported": sum(len(r.claim_unsupported) for r in results),
        # 取"下中位数"（偶数样本时偏小一侧），避免小样本下 P50 被高估
        "latency_p50_ms": latencies[(len(latencies) - 1) // 2] if latencies else 0.0,
        "latency_p95_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0,
        "by_variant": _group_metrics(results, lambda r: r.item.variant or "unlabeled"),
        "by_domain": _group_metrics(results, lambda r: r.item.domain or "unlabeled"),
    }


def _group_metrics(
    results: Sequence[ItemResult],
    key_of: Any,
) -> list[dict[str, Any]]:
    """按 variant / domain 分组重算同一套指标（分组样本小，只给检索与答案口径）。"""
    groups: dict[str, list[ItemResult]] = {}
    for result in results:
        groups.setdefault(str(key_of(result)), []).append(result)
    rows: list[dict[str, Any]] = []
    for name in sorted(groups):
        members = groups[name]
        answerable = [r for r in members if not r.item.should_refuse]
        retrieval = _hit_metrics(members, "rank")
        cited = _hit_metrics(members, "answer_rank")
        points = [point for r in answerable for point in r.point_hits]
        # 整组都是应拒题时 Hit 没有意义，写 None 让渲染层显示 "-"，别留个 0.0 像是全面溃败
        none_when_empty = not answerable
        rows.append(
            {
                "name": name,
                "n": len(members),
                "n_answerable": len(answerable),
                "hit@1": None if none_when_empty else retrieval["hit@1"],
                "hit@3": None if none_when_empty else retrieval["hit@3"],
                "mrr": None if none_when_empty else retrieval["mrr"],
                "answer_hit@1": None if none_when_empty else cited["hit@1"],
                "answer_mrr": None if none_when_empty else cited["mrr"],
                "point_coverage": (
                    sum(1 for r in answerable for found in r.point_hits.values() if found) / len(points)
                    if points
                    else None
                ),
                "refusal_recall": (
                    sum(1 for r in members if r.item.should_refuse and r.refused)
                    / max(1, sum(1 for r in members if r.item.should_refuse))
                    if any(r.item.should_refuse for r in members)
                    else None
                ),
                "false_reject_rate": (
                    sum(1 for r in answerable if r.refused) / len(answerable) if answerable else None
                ),
            }
        )
    return rows


def sweep_thresholds(results: Sequence[ItemResult]) -> list[dict[str, Any]]:
    """基于已记录的 best_score 扫描阈值（只影响闸门 1）。"""
    rows: list[dict[str, Any]] = []
    for threshold in SWEEP_THRESHOLDS:
        answerable = [r for r in results if not r.item.should_refuse]
        refuse_items = [r for r in results if r.item.should_refuse]
        false_reject = sum(1 for r in answerable if r.best_score < threshold)
        correct_refuse = sum(1 for r in refuse_items if r.best_score < threshold)
        rows.append(
            {
                "threshold": threshold,
                "answered": len(answerable) - false_reject,
                "false_reject": false_reject,
                "false_reject_rate": false_reject / len(answerable) if answerable else 0.0,
                "correct_refuse": correct_refuse,
                "refuse_rate": correct_refuse / len(refuse_items) if refuse_items else 0.0,
                "accuracy": (
                    (len(answerable) - false_reject) + correct_refuse
                ) / max(1, len(results)),
            }
        )
    return rows


def recommend_threshold(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """推荐阈值：准确率最高优先，其次误拒率最低，再其次阈值较高（更保守）。"""
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (-row["accuracy"], row["false_reject_rate"], -row["threshold"]),
    )[0]


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_report(
    *,
    label: str,
    metrics: dict[str, Any],
    results: Sequence[ItemResult],
    sweep: Sequence[dict[str, Any]],
    recommended: dict[str, Any] | None,
    environment: dict[str, Any],
    threshold_used: float,
    judge_mode: str,
) -> str:
    lines: list[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# RAG 评测报告 · {label}")
    lines.append("")
    lines.append(f"生成时间：{stamp}")
    lines.append("")
    lines.append("## 1. 环境")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    for key, value in environment.items():
        lines.append(f"| {key} | {value} |")
    lines.append(f"| 生成阈值（实际使用） | {threshold_used} |")
    lines.append(f"| 忠实度判定方式 | {judge_mode} |")
    lines.append("")
    lines.append("## 2. 总体指标")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 题数 | {metrics['n']} | 可答 {metrics['n_answerable']} / 应拒答 {metrics['n_refuse']} |")
    lines.append(f"| Hit@1 | {_fmt(metrics['hit@1'])} | 召回候选中 TOP-1 命中期望文档 |")
    lines.append(f"| Hit@3 | {_fmt(metrics['hit@3'])} | |")
    lines.append(f"| Hit@5 | {_fmt(metrics['hit@5'])} | |")
    lines.append(f"| MRR | {_fmt(metrics['mrr'])} | 期望文档首次出现的名次倒数均值 |")
    lines.append(
        f"| **答案 Hit@1** | {_fmt(metrics['answer_hit@1'])} | "
        "期望文档出现在答案**第 1 个引用**上的比例（口径见 §3.2） |"
    )
    lines.append(f"| **答案 Hit@3** | {_fmt(metrics['answer_hit@3'])} | 同上，放宽到前三个引用 |")
    lines.append(f"| 答案 MRR | {_fmt(metrics['answer_mrr'])} | 期望文档在答案引用序列里的倒数排名均值 |")
    lines.append(f"| 拒答准确率 | {_fmt(metrics['refusal_accuracy'])} | 应拒/应答判断全对的比例 |")
    lines.append(f"| 正确拒答率(recall) | {_fmt(metrics['refusal_recall'])} | 应拒答的题被拒的比例 |")
    lines.append(f"| 拒答精确率 | {_fmt(metrics['refusal_precision'])} | 拒答里确实该拒的比例 |")
    lines.append(f"| 误拒率 | {_fmt(metrics['false_reject_rate'])} | 可答题被拒的比例 = 1 - 可用性 |")
    lines.append(f"| 要点覆盖率 | {_fmt(metrics['point_coverage'])} | expected_answer_points 能在**引用原文**中找到的比例 |")
    lines.append(f"| 忠实度 | {_fmt(metrics['faithfulness'], 3)} | 逐题判定的通过率（null 表示未判定） |")
    if metrics.get("claim_support_rate") is not None:
        lines.append(
            f"| 逐句支持率（claim 校验） | {_fmt(metrics['claim_support_rate'], 3)} | "
            f"句子内容词落在其引用章节原文里的平均比例；本次改标 {metrics['claim_corrections']} 句、"
            f"删句 {metrics['claim_dropped']} 句 |"
        )
    lines.append(f"| 延迟 P50 / P95 | {metrics['latency_p50_ms']:.0f} / {metrics['latency_p95_ms']:.0f} ms | 单题端到端 |")
    lines.append("")
    lines.append("## 3. 逐题结果")
    lines.append("")
    lines.append("| # | 问题 | 应拒 | 实际 | 候选名次 | 答案引 | 最优分 | 引用 | 要点 | 忠实 | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for index, result in enumerate(results, start=1):
        covered = sum(1 for found in result.point_hits.values() if found)
        total_points = len(result.point_hits)
        points = f"{covered}/{total_points}" if total_points else "-"
        lines.append(
            f"| {index} | {result.item.question[:28]} | {'是' if result.item.should_refuse else '否'} "
            f"| {'拒答' if result.refused else '作答'} | {result.rank or '-'} | {result.answer_rank or '-'} "
            f"| {result.best_score:.4f} "
            f"| {len(result.citations)} | {points} | "
            f"{'通过' if result.faithful else ('未判' if result.faithful is None else '不通过')} "
            f"| {result.latency_ms:.0f}ms |"
        )
    lines.append("")
    lines.append("## 3.1 逐句引用校验（闸门 2.5）")
    lines.append("")
    changed = [
        (index, result, kind, item)
        for index, result in enumerate(results, start=1)
        for kind, items in (("改标", result.claim_corrections), ("删句", result.claim_dropped))
        for item in items
    ]
    if not changed:
        lines.append("本次没有需要改标或删除的句子。")
    else:
        lines.append(
            f"共改标 {metrics['claim_corrections']} 句、删句 {metrics['claim_dropped']} 句。"
            "改标 = 原编号的章节找不到依据，但另一章节能对上；“删句”只出现在 strict 模式。"
        )
        lines.append("")
        lines.append("| # | 处理 | 原编号 | 改标为 | 原覆盖 | 新覆盖 | 句子 |")
        lines.append("|---|---|---|---|---|---|---|")
        for index, _result, kind, item in changed:
            before = "".join(f"[{n}]" for n in item.get("markers") or [])
            after = "".join(f"[{n}]" for n in item.get("corrected") or []) or "-"
            lines.append(
                f"| {index} | {kind} | {before or '-'} | {after} | "
                f"{item.get('coverage', 0):.2f} | {item.get('best_coverage', 0):.2f} | "
                f"{str(item.get('claim', ''))[:36]} |"
            )
    lines.append("")
    unsupported = [
        (index, item)
        for index, result in enumerate(results, start=1)
        for item in result.claim_unsupported
    ]
    if unsupported:
        lines.append(
            f"### 仍有 {len(unsupported)} 句未达阈值（fix 模式保留原文，只记诊断）"
        )
        lines.append("")
        lines.append(
            "这些句子在**它所引章节**里找不到足够的词面依据，但也没有另一章节能明显更好，"
            "所以不改标（多属改写型表达或上下文省略）。它们是 `claim_check: strict` 的候选删除对象。"
        )
        lines.append("")
        lines.append("| # | 编号 | 覆盖 | 原因 | 句子 |")
        lines.append("|---|---|---|---|---|")
        for index, item in unsupported[:_MAX_REPORT]:
            markers = "".join(f"[{n}]" for n in item.get("markers") or []) or "-"
            lines.append(
                f"| {index} | {markers} | {item.get('coverage', 0):.2f} | "
                f"{item.get('reason') or '-'} | {str(item.get('claim', ''))[:40]} |"
            )
    lines.append("")
    lines.append("## 3.2 分组指标")
    lines.append("")
    lines.append(
        "两套题分开报是硬要求：`literal` 只换同义词、调语序，主要考生成；"
        "`paraphrase` 换人称换措辞，才是线上用户的问法。混在一起报会被 `literal` 拉高，"
        "看不出短板。`unlabeled` 说明题集还没补 `variant` 字段，别拿它的数字下结论。"
    )
    for title, key, note in (
        ("按改写强度", "by_variant", "题集字段：variant"),
        ("按领域", "by_domain", "题集字段：domain"),
    ):
        rows = metrics.get(key) or []
        if not rows or all(row["name"] == "unlabeled" for row in rows):
            continue
        lines.append("")
        lines.append(f"### {title}（{note}）")
        lines.append("")
        lines.append("| 分组 | 题数 | 可答 | 候选 Hit@1 | 候选 Hit@3 | 答案 Hit@1 | 答案 MRR | 要点 | 误拒率 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['name']} | {row['n']} | {row['n_answerable']} "
                f"| {_fmt(row['hit@1'])} | {_fmt(row['hit@3'])} | {_fmt(row['answer_hit@1'])} "
                f"| {_fmt(row['answer_mrr'])} | {_fmt(row['point_coverage'], 3)} "
                f"| {_fmt(row['false_reject_rate'], 3)} |"
            )
    lines.append("")
    lines.append("## 4. 阈值标定（只影响闸门 1）")
    lines.append("")
    lines.append("| 阈值 | 可答且作答 | 误拒 | 误拒率 | 正确拒答 | 拒答率 | 准确率 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sweep:
        mark = " ⭐" if recommended and row["threshold"] == recommended["threshold"] else ""
        lines.append(
            f"| {row['threshold']:.2f}{mark} | {row['answered']} | {row['false_reject']} | "
            f"{row['false_reject_rate']:.2f} | {row['correct_refuse']} | {row['refuse_rate']:.2f} | "
            f"{row['accuracy']:.2f} |"
        )
    if recommended:
        lines.append("")
        lines.append(
            f"**推荐阈值：{recommended['threshold']:.2f}**（准确率 {recommended['accuracy']:.2f}，"
            f"误拒率 {recommended['false_reject_rate']:.2f}）。"
            "样本只有 10 题，这个值只能当起点，上线前请扩充评测集再标定。"
        )
    lines.append("")
    lines.append("## 5. 详细记录")
    lines.append("")
    for index, result in enumerate(results, start=1):
        lines.append(f"### {index}. {result.item.question}")
        lines.append("")
        lines.append(f"- 期望文档：{result.item.expected_doc or '（应拒答）'}")
        lines.append(f"- 是否拒答：{result.refused}（原因：{result.refusal_reason or '-'}）")
        lines.append(f"- 期望文档名次：{result.rank or '未命中'}")
        lines.append(f"- 判定：{result.judge}｜{result.judge_reason}")
        if result.claim_support_rate is not None:
            lines.append(
                f"- 逐句校验：支持率 {result.claim_support_rate:.2f}（{result.claim_mode}），"
                f"改标 {len(result.claim_corrections)} 句、删句 {len(result.claim_dropped)} 句"
            )
        if result.citations:
            lines.append("- 引用：")
            for citation in result.citations:
                lines.append(
                    f"  - [{citation['index']}]《{citation['doc_name']}》"
                    f"{citation.get('section_path') or '正文'}（{citation.get('page') or '无页码'}）"
                )
        lines.append("")
        lines.append("```text")
        lines.append(result.answer.strip()[:1200])
        lines.append("```")
        lines.append("")
    lines.append("## 6. 口径与限制")
    lines.append("")
    lines.append(
        "- **两套 Hit 口径**：`Hit@K/MRR` 看**召回候选**里有没有 `expected_doc`（只反映检索）；"
        "`答案 Hit@K/答案 MRR` 看**答案正文的引用序列**里期望文档排在第几个 [n]"
        "（闸门 2.5 重编号、上下文预算裁剪、模型的引用选择全部算进成本）。"
        "对外报数字用后者，前者只用来定位是检索还是生成的问题；"
    )
    lines.append("- 要点覆盖与忠实度都基于**被引用的原文**：回答里写了但原文没有 → 不通过；")
    lines.append("- 阈值扫描基于记录下来的 best_score 离线计算，只反映闸门 1（阈值）的影响；")
    lines.append("- 闸门 2（引用校验）与改写质量不在阈值扫描范围内；")
    lines.append(
        "- 逐句支持率是**词面**近似（jieba 内容词覆盖率 + 数字硬校验），不是语义等价判定："
        "它专用来发现“编号错标”与“数字被改写”，改写幅度大的正当回答会被低估；"
        "阈值 0.6 下仍标不过的句子，在 fix 模式下只改标不删句。"
    )
    lines.append("- 若环境表里 `embedder_mock=true`，本报告只验证**链路**，不代表真实语义效果。")
    lines.append(
        "- 语料只有个位数文档时 Hit@K 没有区分度（候选池里本来就没几篇，蒙也对），"
        "这类数字只能当冒烟测试；要对外比，用 `data/bench` 的扩充语料。"
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 配置覆盖（消融表用）
# --------------------------------------------------------------------------
_TRUE_WORDS = {"1", "true", "yes", "on"}
_FALSE_WORDS = {"0", "false", "no", "off"}


def _coerce_like(current: Any, raw: str, dotted: str) -> Any:
    """按**现值类型**解析命令行字符串，避免把 ``reranker.enabled=false`` 变成真字符串。"""
    text = raw.strip()
    if isinstance(current, bool):
        lowered = text.lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise SystemExit(f"--config {dotted} 是布尔项，只能写 true/false（收到 {raw!r}）")
    if isinstance(current, int):
        try:
            return int(text)
        except ValueError as exc:
            raise SystemExit(f"--config {dotted} 需要整数，收到 {raw!r}") from exc
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError as exc:
            raise SystemExit(f"--config {dotted} 需要小数，收到 {raw!r}") from exc
    if isinstance(current, tuple):
        return tuple(part.strip() for part in text.split(",") if part.strip())
    return text


def _set_nested(obj: Any, parts: list[str], raw: str, dotted: str) -> Any:
    """沿点号路径替换叶子字段，返回新的（frozen dataclass 只能用 replace 重建）。"""
    name, *rest = parts
    if not hasattr(obj, name):
        raise SystemExit(f"--config {dotted}：配置里没有 {name!r}，检查拼写（区分大小写、下划线）")
    current = getattr(obj, name)
    if rest:
        return replace(obj, **{name: _set_nested(current, rest, raw, dotted)})
    return replace(obj, **{name: _coerce_like(current, raw, dotted)})


def apply_overrides(settings: Any, overrides: Sequence[str]) -> tuple[Any, list[str]]:
    """把 ``section.key=value`` 应用到 settings，返回 (新对象, 生效清单)。

    消融表每一行都得能一条命令复现，否则数字没法核对；清单会打进报告，
    防止"表里写着 +rerank、实际跑的还是默认配置"这种事。
    """
    applied: list[str] = []
    for text in overrides:
        if "=" not in text:
            raise SystemExit(f"--config 要写成 section.key=value：{text!r}")
        dotted, _, raw = text.partition("=")
        parts = [part.strip() for part in dotted.split(".") if part.strip()]
        if len(parts) < 2:
            raise SystemExit(f"--config {dotted}：至少要有 section.key 两段（例如 retrieval.weight_bm25）")
        settings = _set_nested(settings, parts, raw, dotted)
        applied.append(f"{parts[0]}.{'.'.join(parts[1:])}={raw.strip()}")
    return settings, applied


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def evaluate_item(
    service: ChatService,
    item: GoldenItem,
    *,
    judge_mode: str,
    threshold: float | None = None,
) -> ItemResult:
    """跑一题：检索 + 生成 + 判定。

    ``threshold`` 必须用**实际要上线的阈值**，否则"拒答准确率"测的是假指标
    （用 0 就等于关掉闸门 1）。best_score 会单独记录下来供阈值扫描复用。
    """
    record = service.answerer.answer(item.question, history_text="", threshold=threshold)
    citations, cited_texts = _cited_contexts(record)
    point_hits = _point_coverage(item.expected_answer_points, "\n".join(cited_texts))

    faithful: bool | None = None
    reason = ""
    judge_label = "规则要点覆盖"
    if item.should_refuse:
        faithful = True if record.refused else False
        reason = "应拒答且已拒答" if record.refused else "应拒答但给出了回答"
        judge_label = "拒答判定"
    else:
        if judge_mode == "llm":
            llm_result, llm_reason = _judge_by_llm(service.llm, item, record.answer, cited_texts)
            if llm_result is not None:
                faithful, reason, judge_label = llm_result, llm_reason, "LLM 评审"
        if faithful is None:
            faithful, reason = _judge_by_rule(point_hits, bool(citations))

    return ItemResult(
        item=item,
        answer=record.answer,
        refused=record.refused,
        refusal_reason=record.refusal_reason,
        best_score=record.best_score,
        rank=_rank_of_expected(record, item.acceptable_docs),
        answer_rank=_answer_rank_of_expected(record, item.acceptable_docs),
        citations=citations,
        cited_texts=cited_texts,
        point_hits=point_hits,
        faithful=faithful,
        judge_reason=reason,
        judge=judge_label,
        latency_ms=record.latency_ms,
        usage_prompt=record.usage.prompt_tokens,
        usage_completion=record.usage.completion_tokens,
        claim_support_rate=(record.diagnostics.get("claims") or {}).get("support_rate"),
        claim_mode=str((record.diagnostics.get("claims") or {}).get("mode", "")),
        claim_corrections=list((record.diagnostics.get("claims") or {}).get("corrections") or []),
        claim_dropped=list((record.diagnostics.get("claims") or {}).get("dropped") or []),
        claim_unsupported=list((record.diagnostics.get("claims") or {}).get("unsupported") or []),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG 评测（Hit@K / MRR / 拒答 / 忠实度 / 阈值标定 / 消融）")
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl")
    parser.add_argument("--label", default="run", help="报告标签（写进文件名与标题）")
    parser.add_argument("--threshold", type=float, default=None, help="报告里标注的当前阈值（不影响题目执行）")
    parser.add_argument("--judge", choices=["rule", "llm", "none"], default="rule")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--out", type=Path, default=None, help="报告路径（默认 reports/eval-*.md）")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON 摘要到 stdout")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="SECTION.KEY=值",
        help=(
            "临时覆盖配置，可重复；用于消融表，例如 "
            "--config retrieval.weight_bm25=0 --config reranker.enabled=false"
        ),
    )
    args = parser.parse_args(argv)

    setup_logging()
    settings, applied = apply_overrides(get_settings(), args.config)
    for text in applied:
        print(f"配置覆盖：{text}")
    items = load_golden(args.golden)
    if args.limit:
        items = items[: args.limit]

    service = ChatService(settings)
    try:
        status = service.kb.status()
        if status["vector_chunks"] == 0:
            print("索引为空：请先运行 python scripts\\sync.py --rebuild", file=sys.stderr)
            return 2

        results = [evaluate_item(service, item, judge_mode=args.judge, threshold=args.threshold) for item in items]
        metrics = compute_metrics(results)
        sweep = sweep_thresholds(results)
        recommended = recommend_threshold(sweep)
        threshold_used = settings.retrieval.score_threshold if args.threshold is None else args.threshold

        environment = {
            "LLM": f"{getattr(service.llm, 'model', '?')}（mock={service.llm.is_mock}）",
            "Embedding": f"{service.kb.embedder.name}（mock={service.kb.embedder.is_mock}，dim={service.kb.embedder.dim}）",
            "Reranker": (service.kb.reranker.name if service.kb.reranker else "未启用"),
            "向量库": f"{service.kb.vector_store.name}（{status['vector_chunks']} chunk）",
            "BM25": f"{status['bm25_chunks']} chunk",
            "文档数": status["documents"],
            "collection": status["collection"],
            "评测集": str(args.golden.relative_to(PROJECT_ROOT)) if args.golden.is_relative_to(PROJECT_ROOT) else str(args.golden),
            "配置覆盖": "、".join(applied) if applied else "无（默认配置）",
            "逐句引用校验": (
                f"{settings.generation.claim_check}（阈值 {settings.generation.claim_min_coverage}，"
                f"改标门槛 {settings.generation.claim_reassign_margin}）"
            ),
        }
        judge_mode_label = {"llm": "LLM 评审（失败回落规则）", "rule": "规则要点覆盖", "none": "不做忠实度判定"}[args.judge]
        report = render_report(
            label=args.label,
            metrics=metrics,
            results=results,
            sweep=sweep,
            recommended=recommended,
            environment=environment,
            threshold_used=threshold_used,
            judge_mode=judge_mode_label,
        )

        out_path = args.out or (
            PROJECT_ROOT / "reports" / f"eval-{args.label}-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")

        summary = {
            "label": args.label,
            "metrics": metrics,
            "recommended_threshold": recommended,
            "report": str(out_path),
        }
        print(f"\n报告已写入：{out_path}")
        print(
            f"Hit@1={metrics['hit@1']:.3f} Hit@3={metrics['hit@3']:.3f} Hit@5={metrics['hit@5']:.3f} "
            f"MRR={metrics['mrr']:.3f} 拒答准确率={metrics['refusal_accuracy']:.3f} "
            f"要点覆盖={metrics['point_coverage']:.3f} 忠实度={_fmt(metrics['faithfulness'], 3)}"
        )
        if recommended:
            print(
                f"推荐阈值={recommended['threshold']:.2f}"
                f"（误拒 {recommended['false_reject']} 题，正确拒答 {recommended['correct_refuse']} 题）"
            )
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
