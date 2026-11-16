"""回答文本的两个"去噪"处理。

1. ``flatten_tables``：把资料里的 markdown 表格拍平成"列名：值；列名：值"的句子。
   模型看到 ``|`` 表格会整段照抄进答案（连分隔行 ``|---|`` 一起抄），既难看又
   会把引用摘要撑满；拍平之后它只能按句子引用数值。
2. ``strip_source_appendix``：删掉模型自己写的"引用来源 / 参考资料"清单。
   引用卡片由前端按 citation 渲染，模型再列一份就是重复。
"""

from __future__ import annotations

import re

# 表格：整行都以 | 开头结尾
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# 表头分隔行：| --- | :--: |
_TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

# 模型自带的来源清单标题（容忍 **加粗**、## 标题、冒号/破折号）
_APPENDIX_HEAD = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:引用来源|参考资料|参考来源|引用文献|参考文件|信息来源|来源)"
    r"\s*(?:\*\*)?\s*(?:[:：]|$|[-–—])"
)
_CITATION_HINT = re.compile(r"\[\d{1,2}\]|《")

# 同一行里追加的清单（模型常写成"……按退学处理[1]。  引用来源 - [1] 《学生学籍管理规定.md》…"）
_APPENDIX_INLINE = re.compile(
    r"(?:引用来源|参考资料|参考来源|引用文献|参考文件|信息来源)"
    r"\s*(?:[:：]|[-–—])\s*"
)


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def _render_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    body = rows[1:]
    if not body:
        cells = [cell for cell in header if cell]
        return ["表格：" + "；".join(cells)] if cells else []
    lines: list[str] = []
    for row in body:
        pairs = [f"{name}：{value}" for name, value in zip(header, row) if name and value]
        if pairs:
            lines.append("；".join(pairs))
    if lines:
        return lines
    cells = [cell for cell in header if cell]
    return ["表格：" + "；".join(cells)] if cells else []


def flatten_tables(text: str) -> str:
    """把连续的 markdown 表格行换成每行一条"列名：值"记录。"""
    if "|" not in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    index = 0
    total = len(lines)
    while index < total:
        if not _TABLE_ROW.match(lines[index]):
            out.append(lines[index])
            index += 1
            continue
        rows: list[list[str]] = []
        cursor = index
        while cursor < total and _TABLE_ROW.match(lines[cursor]):
            if not _TABLE_SEP.match(lines[cursor]):
                rows.append(_cells(lines[cursor]))
            cursor += 1
        out.extend(_render_table(rows))
        index = cursor
    return "\n".join(out)


def strip_source_appendix(text: str) -> str:
    """删掉模型在答案末尾自己列的来源清单。"""
    lines = text.splitlines()
    cut = -1
    for position, line in enumerate(lines):
        if _APPENDIX_HEAD.match(line):
            cut = position
            break
    if cut > 0:  # 没有，或整篇就是清单 —— 不动
        tail = "\n".join(lines[cut:])
        if _CITATION_HINT.search(tail):
            head = "\n".join(lines[:cut]).rstrip()
            if len(head) >= 8:
                return head

    # 清单和正文挤在同一行（模型直接接在句号后面）
    match = _APPENDIX_INLINE.search(text)
    if match and match.start() >= 8:
        head = text[: match.start()].rstrip()
        tail = text[match.start() :]
        if _CITATION_HINT.search(tail) and len(head) >= 8:
            return head
    return text
