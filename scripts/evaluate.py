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
from dataclasses import dataclass, field
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
def compute_metrics(results: Sequence[ItemResult]) -> dict[str, Any]:
    answerable = [r for r in results if not r.item.should_refuse]
    refuse_items = [r for r in results if r.item.should_refuse]

    hits = {k: (sum(1 for r in answerable if r.hit_at[k]) / len(answerable)) if answerable else 0.0 for k in HIT_KS}
    reciprocal = [1.0 / r.rank for r in answerable if r.rank]
    mrr = statistics.fmean(reciprocal) if reciprocal else 0.0

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
        "hit@1": hits[1],
        "hit@3": hits[3],
        "hit@5": hits[5],
        "mrr": mrr,
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
    }


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
    lines.append("| # | 问题 | 应拒 | 实际 | 名次 | 最优分 | 引用 | 要点 | 忠实 | 耗时 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for index, result in enumerate(results, start=1):
        covered = sum(1 for found in result.point_hits.values() if found)
        total_points = len(result.point_hits)
        points = f"{covered}/{total_points}" if total_points else "-"
        lines.append(
            f"| {index} | {result.item.question[:28]} | {'是' if result.item.should_refuse else '否'} "
            f"| {'拒答' if result.refused else '作答'} | {result.rank or '-'} | {result.best_score:.4f} "
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
    lines.append("- Hit@K / MRR 以**召回候选**（不是回答正文）里是否出现 `expected_doc` 判定；")
    lines.append("- 要点覆盖与忠实度都基于**被引用的原文**：回答里写了但原文没有 → 不通过；")
    lines.append("- 阈值扫描基于记录下来的 best_score 离线计算，只反映闸门 1（阈值）的影响；")
    lines.append("- 闸门 2（引用校验）与改写质量不在阈值扫描范围内；")
    lines.append(
        "- 逐句支持率是**词面**近似（jieba 内容词覆盖率 + 数字硬校验），不是语义等价判定："
        "它专用来发现“编号错标”与“数字被改写”，改写幅度大的正当回答会被低估；"
        "阈值 0.6 下仍标不过的句子，在 fix 模式下只改标不删句。"
    )
    lines.append("- 若环境表里 `embedder_mock=true`，本报告只验证**链路**，不代表真实语义效果。")
    lines.append("")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description="RAG 评测（Hit@K / MRR / 拒答 / 忠实度 / 阈值标定）")
    parser.add_argument("--golden", type=Path, default=PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl")
    parser.add_argument("--label", default="run", help="报告标签（写进文件名与标题）")
    parser.add_argument("--threshold", type=float, default=None, help="报告里标注的当前阈值（不影响题目执行）")
    parser.add_argument("--judge", choices=["rule", "llm", "none"], default="rule")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--out", type=Path, default=None, help="报告路径（默认 reports/eval-*.md）")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON 摘要到 stdout")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
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
