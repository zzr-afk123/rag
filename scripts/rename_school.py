"""把示例语料里的校名统一换掉（默认 明德学院 → 郑州大学）。

为什么需要脚本：示例文档里有 DOCX/XLSX，直接 grep 找不到（二进制），
用文本编辑器改又容易破坏格式。这里按格式分别处理：

- .md / .txt：纯文本替换；
- .docx：按段落（含表格单元格）替换，保留第一个 run 的样式；
- .xlsx：遍历所有工作表的单元格。

用法::

    python scripts\\rename_school.py                          # 明德学院 → 郑州大学
    python scripts\\rename_school.py --old 明德学院 --to 郑州大学 --dir data/raw
    python scripts\\rename_school.py --dry-run                # 只报告，不写文件

改完记得让索引重新同步（改文件 mtime 会触发增量同步，或手动 POST /kb/sync）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIX = {".md", ".txt", ".jsonl", ".yaml", ".yml"}


def replace_in_runs(paragraph, old: str, new: str) -> int:
    """替换段落文本；跨 run 的情况用"合并到第一个 run"的保守做法。"""
    if old not in paragraph.text:
        return 0
    replaced = paragraph.text.replace(old, new)
    if paragraph.runs:
        paragraph.runs[0].text = replaced
        for run in paragraph.runs[1:]:
            run.text = ""
    else:  # pragma: no cover - 空段落无 run
        paragraph.add_run(replaced)
    return 1


def process_docx(path: Path, old: str, new: str, *, dry: bool) -> int:
    import docx

    document = docx.Document(str(path))
    hits = 0
    for paragraph in document.paragraphs:
        hits += replace_in_runs(paragraph, old, new)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    hits += replace_in_runs(paragraph, old, new)
    for section in document.sections:
        for paragraph in section.header.paragraphs + section.footer.paragraphs:
            hits += replace_in_runs(paragraph, old, new)
    if hits and not dry:
        document.save(str(path))
    return hits


def process_xlsx(path: Path, old: str, new: str, *, dry: bool) -> int:
    import openpyxl

    workbook = openpyxl.load_workbook(str(path))
    hits = 0
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and old in cell.value:
                    cell.value = cell.value.replace(old, new)
                    hits += 1
    if hits and not dry:
        workbook.save(str(path))
    return hits


def process_text(path: Path, old: str, new: str, *, dry: bool) -> int:
    text = path.read_text(encoding="utf-8")
    hits = text.count(old)
    if hits and not dry:
        path.write_text(text.replace(old, new), encoding="utf-8")
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="替换示例语料中的校名")
    parser.add_argument("--old", default="明德学院")
    parser.add_argument("--to", dest="new", default="郑州大学")
    parser.add_argument("--dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == ".docx":
                hits = process_docx(path, args.old, args.new, dry=args.dry_run)
            elif suffix == ".xlsx":
                hits = process_xlsx(path, args.old, args.new, dry=args.dry_run)
            elif suffix in TEXT_SUFFIX:
                hits = process_text(path, args.old, args.new, dry=args.dry_run)
            else:
                continue
        except Exception as exc:  # noqa: BLE001 - 单个文件坏了不影响其余
            print(f"  跳过 {path.name}: {type(exc).__name__}: {exc}")
            continue
        if hits:
            total += hits
            print(f"  {path.name}: {hits} 处")
    print(f"共替换 {total} 处（{args.old} → {args.new}）{'（dry-run，未写文件）' if args.dry_run else ''}")
    if total and not args.dry_run:
        print("提示：索引需要重新同步。服务在跑时会自动增量同步（最长 60s），或手动 POST /kb/sync。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
