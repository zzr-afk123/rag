"""问答日志的统计口径：近 N 天被问了什么、哪些问题没答上、该给哪个部门看。

数据来源只有一处：`logs/qa.jsonl`（含 TimedRotatingFileHandler 轮转出来的
`qa.jsonl.YYYY-MM-DD`）。**不引入数据库**——这个规模（一个学院、一天几十到几百条）
用 JSONL + 内存聚合足够，也便于运维直接 `grep` 排障。

三条口径说明（写在代码里是为了避免以后"数字对不上"的争论）：

1. **按问题的"字面"去重**：去掉空白与中英文标点后再比对。"休学最长多久？"与
   "休学最长多久"算同一个问题；"休学最长可以多久"不算——这是刻意的：
   用它看"哪些话被反复问"，不做语义聚类（那需要模型，成本与不确定性都不值得）。
2. **部门归属优先用日志里的 `departments` 字段**（生成时就已经判定过），
   老日志没有该字段时才用关键词重新匹配。这样口径不会因为后来改了关键词表
   而整体漂移。
3. **时间戳**：`ts` 字段由 `log_qa` 写入；老记录没有 `ts` 时，用轮转文件名里的
   日期（`qa.jsonl.2026-11-10`），再退到文件修改时间。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..departments import Department, DepartmentDirectory
from ..logging_setup import get_logger

logger = get_logger("insights.analytics")

_PUNCT = re.compile(r"[\s\u3000？?！!。，,、；;：:“”\"'（）()【】\[\]…·.]+")
_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})$")


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _fallback_date(path: Path) -> datetime:
    """轮转文件名里的日期优先，其次文件修改时间。"""
    match = _DATE_IN_NAME.search(path.name)
    if match:
        try:
            return datetime.fromisoformat(match.group(1))
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:  # pragma: no cover - 文件刚被删
        return datetime.now()


def load_records(
    log_dir: Path,
    *,
    qa_log: str = "qa.jsonl",
    days: int = 30,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """读取窗口内的问答记录（按时间升序）。坏行跳过，不影响整体。"""
    moment = now or datetime.now()
    since = moment - timedelta(days=max(1, days))
    records: list[dict[str, Any]] = []
    if not Path(log_dir).exists():
        return records

    for path in sorted(Path(log_dir).glob(f"{qa_log}*")):
        if not path.is_file():
            continue
        fallback = _fallback_date(path)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:  # pragma: no cover
            logger.warning("读取问答日志失败：%s（%s）", path, exc)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            stamp = _parse_ts(payload.get("ts")) or fallback
            if stamp < since:
                continue
            payload["_ts"] = stamp.isoformat(timespec="seconds")
            payload["_day"] = stamp.date().isoformat()
            records.append(payload)
    records.sort(key=lambda item: str(item.get("_ts", "")))
    return records


# --------------------------------------------------------------------------
# 归一化与聚合
# --------------------------------------------------------------------------
def normalize_question(text: object) -> str:
    """去空白与标点、统一小写，用来判断"是不是同一个问题"。"""
    if not isinstance(text, str):
        return ""
    return _PUNCT.sub("", text).lower()


def _department_ids(record: Mapping[str, Any], directory: DepartmentDirectory) -> list[str]:
    explicit = record.get("departments")
    if isinstance(explicit, list) and explicit:
        return [str(item) for item in explicit if str(item).strip()]
    directory_hits = directory.match(
        str(record.get("question") or ""),
        rewritten=str(record.get("rewritten_query") or ""),
    )
    return [department.id for department in directory_hits]


def _truncate(text: str, limit: int = 42) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    directory: DepartmentDirectory,
    days: int,
    now: datetime | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """把记录聚合成看板/周报需要的结构。"""
    moment = now or datetime.now()
    since = (moment - timedelta(days=max(1, days))).replace(microsecond=0)

    total = len(records)
    refused = sum(1 for record in records if record.get("refused"))
    latency = [float(record.get("latency_ms") or 0) for record in records if record.get("latency_ms")]
    answered = total - refused

    question_counter: Counter[str] = Counter()
    question_sample: dict[str, str] = {}
    question_refused: Counter[str] = Counter()
    question_depts: dict[str, set[str]] = defaultdict(set)
    per_dept: dict[str, dict[str, Any]] = {}
    unmatched = 0
    daily_total: Counter[str] = Counter()
    daily_refused: Counter[str] = Counter()

    for record in records:
        question = str(record.get("question") or "").strip()
        key = normalize_question(question)
        is_refused = bool(record.get("refused"))
        day = str(record.get("_day") or "")
        if day:
            daily_total[day] += 1
            if is_refused:
                daily_refused[day] += 1

        if key:
            question_counter[key] += 1
            question_sample.setdefault(key, question)
            if is_refused:
                question_refused[key] += 1

        dept_ids = _department_ids(record, directory)
        if not dept_ids:
            unmatched += 1
        for dept_id in dept_ids:
            question_depts[key].add(dept_id)
            bucket = per_dept.setdefault(
                dept_id,
                {
                    "id": dept_id,
                    "name": (directory.by_id(dept_id).name if directory.by_id(dept_id) else dept_id),
                    "count": 0,
                    "refused": 0,
                    "questions": Counter(),
                    "refused_questions": Counter(),
                },
            )
            bucket["count"] += 1
            if is_refused:
                bucket["refused"] += 1
                bucket["refused_questions"][key] += 1
            bucket["questions"][key] += 1

    department_rows: list[dict[str, Any]] = []
    for bucket in sorted(per_dept.values(), key=lambda item: (-item["count"], item["name"])):
        top_questions = [
            {
                "question": question_sample.get(key, key),
                "count": count,
                "refused": question_refused.get(key, 0) > 0,
            }
            for key, count in bucket["questions"].most_common(5)
        ]
        department_rows.append(
            {
                "id": bucket["id"],
                "name": bucket["name"],
                "count": bucket["count"],
                "refused": bucket["refused"],
                "share": round(bucket["count"] / total, 3) if total else 0.0,
                "refusal_rate": round(bucket["refused"] / bucket["count"], 3) if bucket["count"] else 0.0,
                "top_questions": top_questions,
            }
        )

    # 补齐窗口内没有数据的日期，前端折线才不会断（含今天，最多 days 个点）
    daily: list[dict[str, Any]] = []
    end_day = moment.date()
    start_day = max(since.date(), end_day - timedelta(days=max(1, days) - 1))
    cursor_day = start_day
    while cursor_day <= end_day:
        key = cursor_day.isoformat()
        daily.append({"date": key, "total": daily_total.get(key, 0), "refused": daily_refused.get(key, 0)})
        cursor_day += timedelta(days=1)

    top_questions = [
        {
            "question": question_sample.get(key, key),
            "count": count,
            "refused": question_refused.get(key, 0) > 0,
            "departments": sorted(question_depts.get(key, set())),
        }
        for key, count in question_counter.most_common(max(1, top_n))
    ]

    refused_questions = [
        {
            "question": question_sample.get(key, key),
            "count": count,
            "departments": sorted(question_depts.get(key, set())),
        }
        for key, count in question_refused.most_common(max(1, min(top_n, 10)))
    ]

    return {
        "window_days": max(1, days),
        "since": since.isoformat(timespec="seconds"),
        "until": moment.replace(microsecond=0).isoformat(timespec="seconds"),
        "total": total,
        "answered": answered,
        "refused": refused,
        "refusal_rate": round(refused / total, 3) if total else 0.0,
        "unique_questions": len(question_counter),
        "repeat_rate": round(1 - len(question_counter) / total, 3) if total else 0.0,
        "unmatched": unmatched,
        # 演示数据条数（scripts/seed_demo_questions.py 写入的记录带 demo=true）
        "demo": sum(1 for record in records if record.get("demo")),
        "matched_rate": round(1 - unmatched / total, 3) if total else 0.0,
        "avg_latency_ms": round(sum(latency) / len(latency), 1) if latency else 0.0,
        "departments": department_rows,
        "top_questions": top_questions,
        "refused_questions": refused_questions,
        "daily": daily,
        "generated_at": moment.replace(microsecond=0).isoformat(timespec="seconds"),
    }


def dept_slice(summary: Mapping[str, Any], dept_id: str) -> dict[str, Any]:
    """从整体统计里取出单个部门的口径（周报按部门发送时用）。"""
    for row in summary.get("departments", []):
        if row.get("id") == dept_id:
            return dict(row)
    return {}


# --------------------------------------------------------------------------
# 周报 Markdown
# --------------------------------------------------------------------------
def weekly_markdown(
    summary: Mapping[str, Any],
    *,
    department: Department | None = None,
    school_name: str = "",
    site_title: str = "",
    example_data: bool = False,
) -> str:
    """生成一份可直接发邮件/下载的 Markdown 周报。"""
    window = int(summary.get("window_days", 7))
    header = f"# {school_name + ' · ' if school_name else ''}"
    title = f"学生咨询{'周报' if window <= 7 else f'近 {window} 天报告'}"
    if department:
        title += f"（{department.name}）"
    lines: list[str] = [header + title, ""]
    lines.append(f"- 统计窗口：{summary.get('since', '')} ~ {summary.get('until', '')}（{window} 天）")
    lines.append(f"- 生成时间：{summary.get('generated_at', '')}")
    lines.append("")

    if department:
        row = dept_slice(summary, department.id)
        count = int(row.get("count", 0))
        refused = int(row.get("refused", 0))
        total = int(summary.get("total", 0))
        lines.append("## 一、本部门概况")
        lines.append("")
        lines.append(f"- 指向本部门的咨询：**{count}** 条（占全部 {total} 条的 "
                     f"{(count / total * 100 if total else 0):.0f}%）")
        lines.append(f"- 其中未能作答（知识库无依据）：**{refused}** 条")
        lines.append("")
        lines.append("## 二、被问最多的问题")
        lines.append("")
        if row.get("top_questions"):
            lines.append("| # | 次数 | 问题 | 是否未答上 |")
            lines.append("|---|---|---|---|")
            for index, item in enumerate(row["top_questions"], start=1):
                flag = "是" if item.get("refused") else ""
                lines.append(f"| {index} | {item.get('count')} | {_truncate(str(item.get('question', '')))} | {flag} |")
        else:
            lines.append("本期没有指向本部门的咨询。")
        lines.append("")
        lines.append("## 三、建议关注")
        lines.append("")
        for advice in _advice(row):
            lines.append(f"- {advice}")
        lines.append("")
    else:
        lines.append("## 一、总体概况")
        lines.append("")
        lines.append(f"- 咨询总量：**{summary.get('total', 0)}** 条（去重后 {summary.get('unique_questions', 0)} 类问题）")
        lines.append(f"- 未能作答：**{summary.get('refused', 0)}** 条（拒答率 {float(summary.get('refusal_rate', 0)) * 100:.0f}%）")
        lines.append(f"- 能识别到部门的：{float(summary.get('matched_rate', 0)) * 100:.0f}%")
        lines.append("")
        lines.append("## 二、部门分布")
        lines.append("")
        lines.append("| 部门 | 咨询数 | 占比 | 未答上 |")
        lines.append("|---|---|---|---|")
        for row in summary.get("departments", []):
            lines.append(
                f"| {row.get('name')} | {row.get('count')} | {float(row.get('share', 0)) * 100:.0f}% | {row.get('refused')} |"
            )
        lines.append("")
        lines.append("## 三、高频问题")
        lines.append("")
        lines.append("| # | 次数 | 问题 |")
        lines.append("|---|---|---|")
        for index, item in enumerate(summary.get("top_questions", []), start=1):
            lines.append(f"| {index} | {item.get('count')} | {_truncate(str(item.get('question', '')))} |")
        lines.append("")

    if summary.get("refused_questions"):
        lines.append("## 四、被拒答的问题（知识库可能缺料）")
        lines.append("")
        for item in summary["refused_questions"]:
            lines.append(f"- {_truncate(str(item.get('question', '')), 60)}（{item.get('count')} 次）")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("数据来源：问答系统日志（仅统计问题与是否作答，不保存回答正文与个人信息）。")
    if example_data:
        lines.append("")
        lines.append("> 注：联系方式与部分配置为示例数据，请以学校官网公布为准。")
    return "\n".join(lines)


def _advice(row: Mapping[str, Any]) -> list[str]:
    """规则化建议：只讲数据能支撑的话，不做"AI 式"泛泛而谈。"""
    advice: list[str] = []
    count = int(row.get("count", 0) or 0)
    refused = int(row.get("refused", 0) or 0)
    if count and refused / count >= 0.2:
        advice.append(
            f"未答上比例偏高（{refused}/{count}）：建议补充本部门相关制度文件或办事指南，"
            "或把这些问题的标准答复加入知识库。"
        )
    top = row.get("top_questions") or []
    if top and int(top[0].get("count", 0)) >= 2:
        advice.append(f"高频问题「{_truncate(str(top[0].get('question', '')), 30)}」出现 {top[0].get('count')} 次，适合做成置顶说明。")
    if not advice:
        advice.append("本期数据平稳，无需特别处理。")
    return advice


__all__ = ["dept_slice", "load_records", "normalize_question", "summarize", "weekly_markdown"]
