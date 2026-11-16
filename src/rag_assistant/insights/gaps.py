"""知识库缺口清单：把"没答上来的问题"变成"该补哪份文件"。

看板的"需要补料的问题"只是证明"有缺口"；要形成闭环，还差一步——
**把缺口整理成一份可以直接发给部门的清单**：按部门分组、按被问次数排序、
带原始问法与建议，导出成 Markdown 就能贴进邮件或文档。

设计约束：

- 只读 `logs/qa.jsonl` 的聚合结果，**不读取回答正文**（避免带出个人信息）；
- 同一问题的不同问法按 `normalize_question` 归并，但保留最多 3 种原始问法；
- 部门归属复用看板同一套规则（``record["departments"]`` 优先，缺失再走关键词匹配），
  所以"清单里的部门"和"看板上的部门"不会对不上。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Sequence

from ..departments import DepartmentDirectory
from .analytics import _department_ids, _truncate, normalize_question

#: 每条缺口最多保留几种原始问法（超过只留最高频的）
_MAX_WORDINGS = 3
#: 每个部门最多列多少条（清单要能读懂，不是日志倾倒）
_MAX_ITEMS_PER_DEPT = 20
#: 高频阈值：被问这么多次就建议优先补
_HOT_COUNT = 3
_NUMBER_HINT = ("多久", "多少", "几天", "几小时", "几个", "几年", "比例", "金额", "时限")
#: 寒暄/测试类提问拒答了也不算"知识缺口"，列进去只会稀释清单
_SMALL_TALK = {"hi", "hello", "hey", "你好", "您好", "在吗", "在不在", "测试", "test", "1"}


def _hint(question: str, count: int) -> str:
    if count >= _HOT_COUNT:
        return "高频缺口：建议优先补充对应文件或条款"
    if any(token in question for token in _NUMBER_HINT):
        return "涉及具体数值/时限，建议在文件里写清口径"
    return "建议补充该问题对应的制度条款或办事流程"


def collect_gaps(
    records: Sequence[dict[str, Any]],
    *,
    directory: DepartmentDirectory,
    days: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    """把拒答记录聚合成"补料清单"结构。

    Returns:
        含 ``items``（扁平、按次数降序）、``departments``（按部门分组）、
        ``unmatched``（没能定位到部门的部分）与若干统计字段。
    """
    moment = now or datetime.now()
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    ignored = 0

    for record in records:
        if not record.get("refused"):
            continue
        question = str(record.get("question") or "").strip()
        if not question:
            continue
        normalized = normalize_question(question)
        if not normalized:
            continue
        if normalized in _SMALL_TALK or len(normalized) <= 2:
            ignored += 1
            continue
        depts = _department_ids(record, directory)
        dept_id = depts[0] if depts else ""
        entry = groups.get((dept_id, normalized))
        if entry is None:
            entry = {
                "dept_id": dept_id,
                "count": 0,
                "wordings": Counter(),
                "first_seen": "",
                "last_seen": "",
            }
            groups[(dept_id, normalized)] = entry
        entry["count"] += 1
        entry["wordings"][_truncate(question, 60)] += 1
        stamp = str(record.get("_ts") or "")
        if stamp:
            entry["first_seen"] = entry["first_seen"] or stamp
            entry["last_seen"] = stamp

    items: list[dict[str, Any]] = []
    for (dept_id, _normalized), entry in groups.items():
        department = directory.by_id(dept_id) if dept_id else None
        wordings = [text for text, _ in entry["wordings"].most_common(_MAX_WORDINGS)]
        question = wordings[0] if wordings else ""
        items.append(
            {
                "dept_id": dept_id,
                "dept_name": department.name if department else "未识别部门",
                "question": question,
                "count": entry["count"],
                "wordings": wordings[1:],
                "first_seen": entry["first_seen"],
                "last_seen": entry["last_seen"],
                "hint": _hint(question, entry["count"]),
            }
        )
    items.sort(key=lambda item: (item["dept_id"] == "", -item["count"]))

    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        buckets.setdefault(item["dept_id"], []).append(item)

    departments: list[dict[str, Any]] = []
    for dept_id, group in buckets.items():
        if not dept_id:
            continue
        department = directory.by_id(dept_id)
        departments.append(
            {
                "id": dept_id,
                "name": department.name if department else dept_id,
                "full_name": department.full_name if department else "",
                "entries": len(group),
                "count": sum(item["count"] for item in group),
                "items": group[:_MAX_ITEMS_PER_DEPT],
            }
        )
    departments.sort(key=lambda row: (-row["count"], row["id"]))

    unmatched = buckets.get("", [])
    total_refused = sum(item["count"] for item in items)
    return {
        "generated_at": moment.strftime("%Y-%m-%d %H:%M"),
        "window_days": int(days),
        "total_refused": total_refused,
        "unique_refused": len(items),
        "unmatched_count": sum(item["count"] for item in unmatched),
        "ignored_smalltalk": ignored,
        "departments": departments,
        "unmatched": unmatched[:_MAX_ITEMS_PER_DEPT],
        "items": items,
    }


def gaps_markdown(
    report: dict[str, Any],
    *,
    school_name: str = "",
    site_title: str = "",
    example_data: bool = False,
) -> str:
    """把清单渲染成可直接贴进邮件/Markdown 文档的文本。"""
    title = site_title or "制度与办事流程智能问答助手"
    lines: list[str] = []
    lines.append(f"# 制度文件补充清单（近 {report.get('window_days', 0)} 天）")
    lines.append("")
    if school_name:
        lines.append(f"**{school_name}** · {title}")
    else:
        lines.append(f"**{title}**")
    lines.append("")
    lines.append(f"生成时间：{report.get('generated_at', '')}")
    lines.append("")
    lines.append(
        f"窗口内共拒答 **{report.get('total_refused', 0)}** 次，"
        f"去重后 **{report.get('unique_refused', 0)}** 个问题；"
        f"其中 {report.get('unmatched_count', 0)} 次未能定位到部门。"
    )
    ignored = int(report.get("ignored_smalltalk") or 0)
    if ignored:
        lines.append("")
        lines.append(f"另有 {ignored} 次寒暄/测试类提问被拒答，不计入缺口。")
    lines.append("")
    lines.append("> 用法：按部门把下面的问题分派给对应科室，补充文件或条款后")
    lines.append("> 拖到部门看板上传即可入库；问题消失即说明缺口补上了。")
    lines.append("")
    if example_data:
        lines.append("> ⚠️ 当前统计包含演示数据，正式上线前请先清空 `logs/qa.jsonl*`。")
        lines.append("")

    for department in report.get("departments", []):
        lines.append(f"## {department['name']}（{department['count']} 次 / {department['entries']} 个问题）")
        lines.append("")
        for index, item in enumerate(department.get("items", []), start=1):
            lines.append(f"{index}. **{item['question']}** ×{item['count']}")
            if item.get("wordings"):
                lines.append(f"   - 也问成：{'；'.join(item['wordings'])}")
            if item.get("last_seen"):
                lines.append(f"   - 最近一次：{str(item['last_seen']).replace('T', ' ')[:16]}")
            lines.append(f"   - 建议：{item['hint']}")
        lines.append("")

    unmatched = report.get("unmatched") or []
    if unmatched:
        lines.append(f"## 未识别到部门（{report.get('unmatched_count', 0)} 次）")
        lines.append("")
        lines.append("这些问题没能通过关键词定位到部门，建议先确认归属，或补充 `departments.yaml` 的同义词。")
        lines.append("")
        for index, item in enumerate(unmatched, start=1):
            lines.append(f"{index}. **{item['question']}** ×{item['count']}")
            if item.get("wordings"):
                lines.append(f"   - 也问成：{'；'.join(item['wordings'])}")
        lines.append("")

    if not report.get("items"):
        lines.append("窗口内没有拒答记录——暂时不需要补充文件。")
        lines.append("")
    return "\n".join(lines)
