"""周报 Excel 导出（xlsx）。

看板上的"导出周报"以前是 Markdown，发给部门后还得自己贴进表格。这里直接生成
一份带表头样式、冻结首行、能算合计的 xlsx：值班的人打开就能排序筛选，也能据此
填学校的月报表格。

只依赖 openpyxl（项目原本就为 xlsx 解析引入过），不引入新依赖。
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable, Mapping, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .analytics import Department, _advice, dept_slice

BRAND = "0D4C92"
BRAND_LIGHT = "EAF2FB"
GRID = "D8E1EC"
MUTED = "667085"

# 中文表头不能靠自动列宽（openpyxl 不会测量），这里按内容给一个够用的宽度。
_WIDTHS = {
    "指标": 24,
    "数值": 16,
    "说明": 46,
    "部门": 22,
    "咨询数": 12,
    "占比": 12,
    "未答上": 12,
    "未答率": 12,
    "序号": 8,
    "问题": 62,
    "次数": 10,
    "涉及部门": 26,
    "日期": 14,
    "是否未答上": 12,
    "建议": 78,
}

_THIN = Side(style="thin", color=GRID)
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _title(ws, text: str, *, span: int, size: int = 15) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=span)
    cell = ws.cell(row=1, column=1, value=text)
    cell.font = Font(size=size, bold=True, color="1F2A37")
    cell.alignment = Alignment(vertical="center")
    ws.row_dimensions[1].height = 30


def _caption(ws, row: int, text: str) -> None:
    cell = ws.cell(row=row, column=1, value=text)
    cell.font = Font(size=9.5, color=MUTED)


def _header_row(ws, row: int, headers: Sequence[str]) -> None:
    fill = PatternFill("solid", fgColor=BRAND)
    for index, name in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=index, value=name)
        cell.font = Font(bold=True, color="FFFFFF", size=10.5)
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _BORDER
    ws.row_dimensions[row].height = 22


def _body_row(
    ws,
    row: int,
    values: Sequence[Any],
    *,
    total: bool = False,
    percent_cols: Iterable[int] = (),
    wrap_cols: Iterable[int] = (),
) -> None:
    percent_cols = set(percent_cols)
    wrap_cols = set(wrap_cols)
    fill = PatternFill("solid", fgColor=BRAND_LIGHT) if total else None
    for index, value in enumerate(values, start=1):
        cell = ws.cell(row=row, column=index, value=value)
        cell.border = _BORDER
        cell.font = Font(size=10.5, bold=total)
        if fill is not None:
            cell.fill = fill
        if index in percent_cols and isinstance(value, (int, float)):
            cell.number_format = "0.0%"
            cell.alignment = Alignment(horizontal="right")
        elif isinstance(value, (int, float)):
            cell.alignment = Alignment(horizontal="right")
        elif index in wrap_cols:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        else:
            cell.alignment = Alignment(vertical="top")


def _widths(ws, headers: Sequence[str]) -> None:
    for index, name in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(index)].width = _WIDTHS.get(name, 16)


def _kpi_sheet(
    wb: Workbook,
    summary: Mapping[str, Any],
    *,
    department: Department | None,
    school_name: str,
    example_data: bool,
    dept_names: Mapping[str, str] | None,
) -> None:
    ws = wb.active
    ws.title = "概况"
    window = int(summary.get("window_days", 7) or 0)
    label = "学生咨询周报" if window <= 7 else f"学生咨询近 {window} 天报告"
    if department is not None:
        label += f"（{department.name}）"
    prefix = f"{school_name} · " if school_name else ""
    _title(ws, prefix + label, span=3)
    _caption(ws, 2, f"统计窗口：{summary.get('since', '')} ~ {summary.get('until', '')}（{window} 天）")
    _caption(ws, 3, f"生成时间：{summary.get('generated_at', '')}")

    headers = ("指标", "数值", "说明")
    _header_row(ws, 5, headers)
    if department is not None:
        row = dept_slice(summary, department.id)
        values = [
            ("指向本部门的咨询", int(row.get("count", 0)), "按关键词归属到本部门的提问次数"),
            ("占全部咨询", float(row.get("share", 0.0)), "本部门咨询数 / 统计窗口内全部咨询数"),
            ("其中未能作答", int(row.get("refused", 0)), "知识库里找不到依据的条数"),
            ("本部门未答率", float(row.get("refusal_rate", 0.0)), "未能作答 / 指向本部门的咨询"),
            ("全库咨询总量", int(summary.get("total", 0)), "同一问题重复提问按次计"),
        ]
    else:
        values = [
            ("咨询总量", int(summary.get("total", 0)), "同一问题重复提问按次计"),
            ("去重后问题数", int(summary.get("unique_questions", 0)), "去掉标点与空白后的去重提问"),
            ("未能作答", int(summary.get("refused", 0)), "知识库里找不到依据、明确拒答的条数"),
            ("拒答率", float(summary.get("refusal_rate", 0.0)), "未能作答 / 咨询总量"),
            ("可识别部门比例", float(summary.get("matched_rate", 0.0)), "按关键词命中归属到部门的比例"),
            ("平均响应", round(float(summary.get("avg_latency_ms", 0) or 0) / 1000, 2), "秒；日志记录的检索 + 生成耗时"),
        ]
    row_index = 6
    for name, value, note in values:
        _body_row(
            ws,
            row_index,
            (name, value, note),
            percent_cols=(2,) if name in {"拒答率", "占全部咨询", "本部门未答率", "可识别部门比例"} else (),
            wrap_cols=(3,),
        )
        row_index += 1

    row_index += 1
    _caption(ws, row_index, "数据来源：问答系统日志（只统计问题与是否作答，不保存回答正文与个人信息）。")
    row_index += 1
    if department is not None:
        _caption(ws, row_index, "部门归属按关键词命中判定，一条提问可能同时属于多个部门。")
        row_index += 1
    if example_data:
        _caption(ws, row_index, "注：联系方式与部分配置为示例数据，请以学校官网公布为准。")
    _widths(ws, headers)
    ws.freeze_panes = "A6"


def _dept_sheet(
    wb: Workbook, summary: Mapping[str, Any], *, dept_names: Mapping[str, str] | None
) -> None:
    ws = wb.create_sheet("部门分布")
    headers = ("部门", "咨询数", "占比", "未答上", "未答率")
    _title(ws, "各部门咨询分布", span=len(headers))
    _caption(ws, 2, "点开可以按咨询数排序；占比为该部门咨询数 / 咨询总量。")
    _header_row(ws, 4, headers)
    rows = summary.get("departments", []) or []
    total = int(summary.get("total", 0) or 0)
    refused_total = 0
    index = 5
    for item in rows:
        count = int(item.get("count", 0) or 0)
        refused = int(item.get("refused", 0) or 0)
        refused_total += refused
        _body_row(
            ws,
            index,
            (
                str(item.get("name") or (dept_names or {}).get(str(item.get("id")), "") or ""),
                count,
                float(item.get("share", 0.0) or 0.0),
                refused,
                (refused / count) if count else 0.0,
            ),
            percent_cols=(3, 5),
        )
        index += 1
    if rows:
        _body_row(
            ws,
            index,
            ("合计", total, 1.0 if total else 0.0, refused_total, (refused_total / total) if total else 0.0),
            total=True,
            percent_cols=(3, 5),
        )
    else:
        _body_row(ws, index, ("本期没有可归属到部门的咨询", "", "", "", ""))
    _widths(ws, headers)
    ws.freeze_panes = "A5"


def _questions_sheet(wb: Workbook, summary: Mapping[str, Any]) -> None:
    ws = wb.create_sheet("高频问题")
    headers = ("序号", "问题", "次数")
    _title(ws, "被问最多的问题", span=len(headers))
    _header_row(ws, 3, headers)
    index = 4
    for order, item in enumerate(summary.get("top_questions", []) or [], start=1):
        _body_row(ws, index, (order, str(item.get("question", "")), int(item.get("count", 0) or 0)), wrap_cols=(2,))
        index += 1
    if index == 4:
        _body_row(ws, index, ("本期没有咨询记录。", "", ""))
    _widths(ws, headers)
    ws.freeze_panes = "A4"


def _refused_sheet(wb: Workbook, summary: Mapping[str, Any], *, dept_names: Mapping[str, str] | None) -> None:
    items = summary.get("refused_questions", []) or []
    ws = wb.create_sheet("未答上的问题")
    headers = ("序号", "问题", "次数", "涉及部门")
    _title(ws, "未能作答的问题（知识库可能缺料）", span=len(headers))
    _caption(ws, 2, "同一份清单也可以在部门看板点「导出补料清单」下载，按部门分派更直接。")
    _header_row(ws, 4, headers)
    index = 5
    for order, item in enumerate(items, start=1):
        ids = item.get("departments") or []
        names = "、".join((dept_names or {}).get(str(i), str(i)) for i in ids) or "未识别"
        _body_row(
            ws,
            index,
            (order, str(item.get("question", "")), int(item.get("count", 0) or 0), names),
            wrap_cols=(2, 4),
        )
        index += 1
    if index == 5:
        _body_row(ws, index, ("本期没有拒答记录。", "", "", ""))
    _widths(ws, headers)
    ws.freeze_panes = "A5"


def _daily_sheet(wb: Workbook, summary: Mapping[str, Any]) -> None:
    rows = summary.get("daily", []) or []
    ws = wb.create_sheet("每日趋势")
    headers = ("日期", "咨询数", "未答上", "未答率")
    _title(ws, "按天的咨询量", span=len(headers))
    _header_row(ws, 3, headers)
    index = 4
    for item in rows:
        total = int(item.get("total", 0) or 0)
        refused = int(item.get("refused", 0) or 0)
        _body_row(
            ws,
            index,
            (str(item.get("date", "")), total, refused, (refused / total) if total else 0.0),
            percent_cols=(4,),
        )
        index += 1
    if index == 4:
        _body_row(ws, index, ("本期没有咨询记录。", "", "", ""))
    _widths(ws, headers)
    ws.freeze_panes = "A4"


def _advice_sheet(wb: Workbook, summary: Mapping[str, Any], department: Department) -> None:
    row = dept_slice(summary, department.id)
    ws = wb.create_sheet("建议关注")
    headers = ("序号", "建议")
    _title(ws, "建议关注", span=len(headers))
    _header_row(ws, 3, headers)
    index = 4
    for order, text in enumerate(_advice(row), start=1):
        _body_row(ws, index, (order, text), wrap_cols=(2,))
        index += 1
    _widths(ws, headers)


def weekly_workbook(
    summary: Mapping[str, Any],
    *,
    department: Department | None = None,
    school_name: str = "",
    site_title: str = "",
    example_data: bool = False,
    dept_names: Mapping[str, str] | None = None,
) -> bytes:
    """生成周报 xlsx 的字节流（可直接作为附件返回）。

    ``department`` 为空时是全校汇总（概况 + 部门分布 + 高频问题 + 未答上 + 每日趋势），
    给出部门时换成本部门视角（概况 + 高频问题 + 建议关注）。
    """
    wb = Workbook()
    _kpi_sheet(
        wb,
        summary,
        department=department,
        school_name=school_name,
        example_data=example_data,
        dept_names=dept_names,
    )
    if department is not None:
        ws = wb.create_sheet("高频问题")
        headers = ("序号", "问题", "次数", "是否未答上")
        row = dept_slice(summary, department.id)
        _title(ws, f"{department.name} · 被问最多的问题", span=len(headers))
        _header_row(ws, 3, headers)
        index = 4
        for order, item in enumerate(row.get("top_questions", []) or [], start=1):
            _body_row(
                ws,
                index,
                (order, str(item.get("question", "")), int(item.get("count", 0) or 0), "是" if item.get("refused") else ""),
                wrap_cols=(2,),
            )
            index += 1
        if index == 4:
            _body_row(ws, index, ("本期没有指向本部门的咨询。", "", "", ""))
        _widths(ws, headers)
        ws.freeze_panes = "A4"
        _advice_sheet(wb, summary, department)
    else:
        _dept_sheet(wb, summary, dept_names=dept_names)
        _questions_sheet(wb, summary)
        _refused_sheet(wb, summary, dept_names=dept_names)
        _daily_sheet(wb, summary)

    wb.properties.title = f"{site_title or school_name} 学生咨询周报"
    wb.properties.creator = "制度与办事流程智能问答助手"
    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


__all__ = ["weekly_workbook"]
