"""XLSX 解析器：每个 sheet 视为一张独立的表格。

制度类 Excel 几乎都是"标准表"（报销标准、材料清单），因此**不拆行**：
一个 sheet 就是一个 :class:`TableBlock`，切块阶段若超长才会按行拆分并重复表头。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..models import Section
from ...exceptions import ParseError
from .base import BaseParser, register_parser

_MAX_ROWS_PER_SHEET = 5000


def _cell_to_text(value) -> str:
    """单元格值转字符串，日期/布尔值做可读化处理。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dt.datetime, dt.date)):
        return value.strftime("%Y-%m-%d") if isinstance(value, dt.date) else value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().replace("\n", " ")


@register_parser
class XlsxParser(BaseParser):
    """把每个工作表解析为一个章节 + 一张表。"""

    name = "xlsx"
    extensions = ("xlsx", "xlsm")

    def parse_sections(self, path: Path) -> list[Section]:
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover
            raise ImportError("解析 XLSX 需要 openpyxl：pip install openpyxl") from exc

        try:
            workbook = openpyxl.load_workbook(filename=str(path), read_only=True, data_only=True)
        except Exception as exc:  # noqa: BLE001 - 损坏 zip / 非 xlsx 都归为解析失败
            raise ParseError(f"XLSX 打开失败（可能已损坏或不是 xlsx）：{path.name} -> {exc}") from exc
        sections: list[Section] = []
        try:
            for sheet in workbook.worksheets:
                rows: list[list[str]] = []
                for index, row in enumerate(sheet.iter_rows(values_only=True)):
                    if index >= _MAX_ROWS_PER_SHEET:
                        break
                    cells = [_cell_to_text(v) for v in row]
                    if any(cells):
                        rows.append(cells)
                if not rows:
                    continue
                title = f"工作表：{sheet.title}"
                section = Section(path=(title,), level=1, page=None)
                # 不给 caption：章节路径已经带工作表名，再写一遍会在引用里重复
                section.add_table(rows)
                sections.append(section)
        finally:
            workbook.close()
        return sections


__all__ = ["XlsxParser"]
