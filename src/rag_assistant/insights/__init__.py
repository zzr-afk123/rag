"""统计与周报：把问答日志变成"部门能看、能用"的东西。

模块划分（刻意很小）：
- ``analytics``：读 `logs/qa.jsonl` → 聚合结构 / 周报 Markdown，纯函数、无 IO 副作用；
- ``mailer``：把周报渲染成邮件，按部门投递（默认只落盘 `.eml`，不真发）；
- ``scheduler``：每周固定时刻触发一次投递（默认不启用）。

分发逻辑与问答链路完全解耦：看板（`/dept`）读的是同一份聚合结果，
所以"页面上看到的数字"和"邮件里发出去的数字"必然一致。
"""

from __future__ import annotations

from .analytics import dept_slice, load_records, normalize_question, summarize, weekly_markdown
from .gaps import collect_gaps, gaps_markdown
from .mailer import Delivery, build_email, deliver_weekly
from .scheduler import WeeklyReportScheduler
from .xlsx import weekly_workbook

__all__ = [
    "Delivery",
    "WeeklyReportScheduler",
    "build_email",
    "collect_gaps",
    "deliver_weekly",
    "dept_slice",
    "gaps_markdown",
    "load_records",
    "normalize_question",
    "summarize",
    "weekly_markdown",
    "weekly_workbook",
]
