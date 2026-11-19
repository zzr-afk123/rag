"""运维台的数据加工：回答质量、错误日志、目录体积。

和 `analytics.py` 的分工：
- `analytics` 面向**部门**，回答"学生都在问什么、哪个部门该补什么料"；
- 这里面向**运维/维护人员**，回答"服务现在准不准、有没有报错、索引多大"。

两者都只读 `logs/qa.jsonl` 与 `logs/app.log`，不碰回答正文。
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

#: 拒答原因的对外中文说明（没列到的直接显示原始值，避免日志里出现新原因时被吞掉）
REASON_LABELS: dict[str, str] = {
    "off_topic": "问题不在知识库范围",
    "below_threshold": "相关度低于门槛",
    "no_candidates": "没有召回任何片段",
    "no_context": "资料块为空",
    "claim_unsupported": "逐句校验未通过",
    "citation_invalid": "引用编号对不上原文",
    "llm_refused": "模型判断资料不足",
    "llm_error": "生成服务异常",
    "empty_answer": "模型返回空答案",
    "empty_question": "空问题",
    "unknown": "未记录原因",
}

#: 日志行的级别字段，用于从 app.log 里挑错误
_LOG_LINE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})[^\|]*\|\s*(?P<level>[A-Z]+)\s*\|")

#: "最近 N 次"耗时统计的样本数
RECENT_SAMPLE = 50


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return round(ordered[index], 1)


def _recent_latency(values: list[float], sample: int) -> dict[str, Any]:
    """最近若干次作答的耗时。

    整窗的 P95 会被早期慢样本（模型/重排还没调好时留下的几百秒记录）拉高，
    看板上同时给出"最近 N 次"更能反映当前手感。
    """
    tail = values[-sample:] if sample > 0 else []
    return {
        "p50": _percentile(tail, 0.5),
        "p95": _percentile(tail, 0.95),
        "max": round(max(tail), 1) if tail else 0.0,
        "samples": len(tail),
    }


def quality_report(
    records: Iterable[dict[str, Any]],
    *,
    days: int,
    now: datetime | None = None,
    top_n: int = 6,
) -> dict[str, Any]:
    """从问答日志算"回答质量"：应答率、引用覆盖、拒答原因、耗时分布。

    口径说明（页面上也会展示）：
    - **应答率** = 有作答次数 / 总次数；它不等于"答得对"，只是"没有拒答"；
    - **引用覆盖率** = 带引用的作答 / 作答次数，衡量"每句都有出处"这条底线有没有守住；
    - **逐句支持率** 来自回答时的 claim 校验，只在开启逐句校验的版本里才有样本。
    """
    moment = now or datetime.now()
    start = moment - timedelta(days=max(1, days))

    total = 0
    answered = 0
    refused = 0
    with_citation = 0
    scores: list[float] = []
    latencies: list[float] = []
    supports: list[float] = []
    reasons: Counter[str] = Counter()
    refused_questions: Counter[str] = Counter()
    daily: dict[str, dict[str, int]] = {}

    for record in records:
        ts = _parse_ts(record.get("ts") or record.get("_ts"))
        if ts is not None and ts < start:
            continue
        total += 1
        day = (ts or moment).date().isoformat()
        bucket = daily.setdefault(day, {"total": 0, "refused": 0})
        bucket["total"] += 1

        is_refused = bool(record.get("refused"))
        if is_refused:
            refused += 1
            bucket["refused"] += 1
            reason = str(record.get("refusal_reason") or "unknown")
            reasons[reason] += 1
            question = str(record.get("question") or "").strip()
            if question:
                refused_questions[question] += 1
        else:
            answered += 1
            if record.get("retrieved_ids"):
                with_citation += 1

        score = record.get("best_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        support = record.get("claims_support")
        if isinstance(support, (int, float)):
            supports.append(float(support))

    days_span = [
        {
            "date": (moment - timedelta(days=offset)).date().isoformat(),
            "total": 0,
            "refused": 0,
        }
        for offset in range(max(0, days - 1), -1, -1)
    ]
    for row in days_span:
        row.update(daily.get(row["date"], {"total": 0, "refused": 0}))

    reason_rows = [
        {
            "reason": reason,
            "label": REASON_LABELS.get(reason, reason),
            "count": count,
            "share": round(count / refused, 3) if refused else 0.0,
        }
        for reason, count in reasons.most_common(top_n)
    ]

    return {
        "window_days": days,
        "generated_at": moment.isoformat(timespec="seconds"),
        "total": total,
        "answered": answered,
        "refused": refused,
        "answer_rate": round(answered / total, 3) if total else 0.0,
        "refusal_rate": round(refused / total, 3) if total else 0.0,
        "with_citation": with_citation,
        "citation_rate": round(with_citation / answered, 3) if answered else 0.0,
        "avg_best_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "claims_samples": len(supports),
        "avg_claims_support": round(sum(supports) / len(supports), 3) if supports else 0.0,
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 1) if latencies else 0.0,
            "samples": len(latencies),
        },
        "latency_recent": _recent_latency(latencies, RECENT_SAMPLE),
        "reasons": reason_rows,
        "top_refused": [
            {"question": question, "count": count}
            for question, count in refused_questions.most_common(top_n)
        ],
        "daily": days_span,
    }


def read_tail(path: Path, *, max_bytes: int = 256 * 1024) -> str:
    """只读文件尾部，避免几十 MB 的日志整个读进内存。"""
    if not path.exists() or not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def recent_events(
    path: Path,
    *,
    levels: tuple[str, ...] = ("ERROR", "CRITICAL", "WARNING"),
    limit: int = 30,
) -> list[dict[str, Any]]:
    """从应用日志尾部挑出错误/告警行，给运维台"最近发生了什么"。

    同一条告警会随每次重启重复出现（例如"账号仍在使用示例口令"），
    这里按（级别, 正文）合并，只留最新一次并给出次数，免得面板被刷屏。
    """
    wanted = {level.upper() for level in levels}
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for line in read_tail(path).splitlines():
        text = line.strip()
        if not text:
            continue
        match = _LOG_LINE.match(text)
        if match is None:
            continue
        level = match.group("level").upper()
        if level not in wanted:
            continue
        message = text[match.end() :].strip()[:400]
        key = (level, message)
        current = merged.get(key)
        if current is None:
            merged[key] = {"ts": match.group("ts"), "level": level, "message": message, "count": 1}
        else:
            current["count"] += 1
            current["ts"] = max(current["ts"], match.group("ts"))
    rows = sorted(merged.values(), key=lambda row: row["ts"], reverse=True)
    return rows[:limit]


def dir_size(path: Path) -> int:
    """目录（或单个文件）占用的字节数；用于"索引多大、日志多大"。"""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def human_size(num: float) -> str:
    """把字节数写成 1.2 MB 这种形式。"""
    value = float(num or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def latest_eval_summary(reports_dir: Path) -> dict[str, Any] | None:
    """找最近一份评测报告里的关键数字，供运维台展示"离线准确率"。

    评测报告是 Markdown（`reports/eval-*.md`），这里只按行取几项固定指标，
    找不到就返回 None——页面会显示"还没有离线评测"，而不是编一个数字出来。
    """
    if not reports_dir.exists():
        return None
    candidates = sorted(
        (path for path in reports_dir.glob("eval-*.md") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    patterns = {
        "decision_accuracy": r"判断准确率\D*([0-9.]+%)",
        "induced_rate": r"被诱导\D*([0-9.]+%)",
        "doc_hit_rate": r"文档命中\D*([0-9.]+%)",
        "point_coverage": r"要点覆盖\D*([0-9.]+%)",
        "hit_at_1": r"Hit@1\D*([0-9./]+)",
    }
    # 报告目录里既有对抗集摘要（有这几项指标），也有逐题明细（没有），
    # 取"最新的一份且至少能解析出一项指标"，否则页面会显示成"没解析到指标"。
    for latest in candidates[:5]:
        text = read_tail(latest, max_bytes=64 * 1024)
        numbers: dict[str, Any] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                numbers[key] = match.group(1)
        if not numbers:
            continue
        return {
            "file": latest.name,
            "mtime": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(timespec="seconds"),
            "metrics": numbers,
        }
    return None


def json_number(value: object) -> float | None:
    """日志里偶尔会写成字符串的数字，统一转成 float。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


__all__ = [
    "REASON_LABELS",
    "dir_size",
    "human_size",
    "json_number",
    "latest_eval_summary",
    "quality_report",
    "read_tail",
    "recent_events",
]
