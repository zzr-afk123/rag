r"""校验 `data/bench` 的合成语料与题集（题写得不对，指标全是噪声）。

为什么需要这个脚本：语料和题目是分批、可能由不同人（或模型）写的，而 Hit@K、
要点覆盖率这两个口径**完全依赖题集与原文逐字对应**。只要出现下面任何一种情况，
跑出来的数字就没有意义，而且不会报错、只会静默变难看：

- `expected_doc` 写成不存在的文件名 → 该题永远判检索失败；
- `expected_answer_points` 是"意译"而不是原文 → 要点覆盖率恒 0；
- `expected_sections` 与切块器实际产出的条标题不一致 → 章节级诊断错位；
- 某个答案要点在**别的文档**里也逐字出现 → 答案 Hit 会误判成"检索对了"；
- 某篇文档从来没被任何题目引用 → 白白增大候选池，把所有指标压低。

用法::

    python scripts/bench_check.py                      # 校验全部 golden_*.jsonl
    python scripts/bench_check.py --json               # 给 CI/脚本消费
    python scripts/bench_check.py --allow-unlabeled    # 还没补 variant/domain 时先跑通

退出码：0 全部通过；1 有 error；2 目录或文件找不到。warning 不影响退出码。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCH_DIR = PROJECT_ROOT / "data" / "bench"
VARIANTS = {"literal", "paraphrase"}
#: SPEC 约定的篇幅区间（太短记不住章节、太长一个 chunk 装不下）
DOC_MIN_CHARS = 1200
DOC_MAX_CHARS = 2200

CHAPTER_RE = re.compile(r"^## (第[一二三四五六七八九十百]+章.*)$", re.M)
ARTICLE_RE = re.compile(r"^\*\*(第[一二三四五六七八九十百]+条[^\*]*)\*\*$", re.M)
#: 题集里的要点/章节常见的全角空格、不间断空格，比对前先归一
SPACE_RE = re.compile(r"[\s\u3000\xa0]+")


def normalize(text: str) -> str:
    """去掉所有空白并统一全角/半角，用于"逐字命中"判定。

    评测脚本 `_point_coverage` 也是按"去空格后是否出现"判的，这里必须保持同一口径，
    否则校验脚本说 OK、评测却说找不到，来回折腾一次比谁都惨。
    """
    return SPACE_RE.sub("", unicodedata.normalize("NFKC", text))


def cn_number_to_int(text: str) -> int | None:
    """把"第七条"里的中文数字转成 7，用于检查条款序号有没有断档。"""
    digits = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    match = re.match(r"第([一二三四五六七八九十百]+)条", text)
    if not match:
        return None
    raw = match.group(1)
    if raw in digits:
        return digits[raw]
    if raw.startswith("十"):
        return 10 + (digits.get(raw[1:], 0) if len(raw) > 1 else 0)
    if len(raw) == 2 and raw[0] in digits and raw[1] == "十":
        return digits[raw[0]] * 10
    if len(raw) == 3 and raw[0] in digits and raw[1] == "十" and raw[2] in digits:
        return digits[raw[0]] * 10 + digits[raw[2]]
    return None


class Checker:
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.docs: dict[str, str] = {}
        self.norm: dict[str, str] = {}
        self.articles: dict[str, set[str]] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, Any] = {}

    # ---- 语料 ----
    def load_docs(self) -> None:
        paths = sorted(self.raw_dir.glob("*.md"))
        if not paths:
            raise SystemExit(f"找不到语料：{self.raw_dir} 里没有 .md")
        for path in paths:
            text = path.read_text(encoding="utf-8")
            self.docs[path.name] = text
            self.norm[path.name] = normalize(text)
            self.articles[path.name] = set(ARTICLE_RE.findall(text))
            self._check_doc_shape(path.name, text)

    def _check_doc_shape(self, name: str, text: str) -> None:
        length = len(normalize(text))
        if not DOC_MIN_CHARS <= length <= DOC_MAX_CHARS:
            self.warnings.append(f"{name}：篇幅 {length} 字不在 {DOC_MIN_CHARS}-{DOC_MAX_CHARS} 区间")
        chapters = CHAPTER_RE.findall(text)
        articles = ARTICLE_RE.findall(text)
        if len(chapters) < 3:
            self.errors.append(f"{name}：只有 {len(chapters)} 个 `## 第X章` 标题，切块器产不出章节路径")
        if len(articles) < 8:
            self.errors.append(f"{name}：只有 {len(articles)} 个 `**第X条 …**` 条标题，章节粒度太粗")
        numbers = [value for value in (cn_number_to_int(a) for a in articles) if value]
        if len(numbers) != len(articles):
            self.warnings.append(f"{name}：有条标题序号不是\"第N条\"写法，跳过连续性检查")
        elif numbers != list(range(1, len(numbers) + 1)):
            self.errors.append(f"{name}：条款序号不连续（{numbers}），题集引用会对不上")
        heads = {normalize(a) for a in articles}
        if len(heads) != len(articles):
            self.errors.append(f"{name}：有条标题重复，同名章节会让引用歧义")

    # ---- 题目 ----
    def check_goldens(self, files: list[Path], *, allow_unlabeled: bool, min_doc_coverage: float) -> None:
        seen_questions: dict[str, str] = {}
        used_docs: Counter[str] = Counter()
        items_total = 0
        variant_counter: Counter[str] = Counter()
        for file in files:
            for number, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                where = f"{file.name}:{number}"
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.errors.append(f"{where}：JSON 解析失败（{exc}）")
                    continue
                items_total += 1
                self._check_item(where, item, seen_questions, used_docs, variant_counter, allow_unlabeled)

        referenced = sum(1 for name in self.docs if used_docs[name])
        coverage = referenced / len(self.docs) if self.docs else 0.0
        # 没人问的文档不是 bug，反而增加了检索难度；但大面积没人问说明题集还没写完，
        # 这时候跑出来的指标是半成品，必须拦下来。
        if coverage < min_doc_coverage:
            self.errors.append(
                f"只有 {referenced}/{len(self.docs)} 篇文档被题目引用（覆盖率 {coverage:.0%} < "
                f"{min_doc_coverage:.0%}），题集没写全就别看指标"
            )
        for name in sorted(self.docs):
            if used_docs[name] == 0:
                self.warnings.append(f"{name}：没有题目引用它")

        self.stats = {
            "docs": len(self.docs),
            "docs_referenced": referenced,
            "doc_coverage": round(coverage, 3),
            "items": items_total,
            "by_doc": dict(used_docs),
            "by_variant": dict(variant_counter),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    def _check_item(
        self,
        where: str,
        item: dict[str, Any],
        seen_questions: dict[str, str],
        used_docs: Counter[str],
        variant_counter: Counter[str],
        allow_unlabeled: bool,
    ) -> None:
        question = str(item.get("question", "")).strip()
        if not question:
            self.errors.append(f"{where}：缺 question")
            return
        if question in seen_questions:
            self.errors.append(f"{where}：问题与 {seen_questions[question]} 完全重复")
        else:
            seen_questions[question] = where

        variant = str(item.get("variant", "")).strip()
        if variant:
            if variant not in VARIANTS:
                self.errors.append(f"{where}：variant={variant!r} 不是 {sorted(VARIANTS)} 之一")
            else:
                variant_counter[variant] += 1
        elif not allow_unlabeled:
            self.errors.append(f"{where}：缺 variant（literal/paraphrase），两套题没法分开报")
        domain = str(item.get("domain", "")).strip()
        if not domain and not allow_unlabeled:
            self.errors.append(f"{where}：缺 domain")

        docs = [str(doc) for doc in (item.get("expected_docs") or []) if str(doc).strip()]
        if item.get("expected_doc"):
            docs.append(str(item["expected_doc"]))
        docs = list(dict.fromkeys(docs))

        if item.get("should_refuse"):
            if docs:
                self.errors.append(f"{where}：should_refuse 的题不该有 expected_doc")
            if not str(item.get("note", "")).strip():
                self.warnings.append(f"{where}：拒答题没写 note 说明为什么该拒")
            return
        if not docs:
            self.errors.append(f"{where}：可答题缺 expected_doc/expected_docs")
            return
        missing = [doc for doc in docs if doc not in self.docs]
        if missing:
            self.errors.append(f"{where}：expected_doc 不存在：{missing}")
            return
        used_docs.update(docs)

        points = [str(point) for point in (item.get("expected_answer_points") or []) if str(point).strip()]
        if not points:
            self.errors.append(f"{where}：可答题缺 expected_answer_points，要点覆盖率会恒 0")
        accepted = "|".join(self.norm[doc] for doc in docs)
        unique_points = 0
        for point in points:
            needle = normalize(point)
            if needle not in accepted:
                self.errors.append(f"{where}：要点「{point}」在 {docs} 原文里逐字找不到")
                continue
            clashes = [name for name in self.docs if name not in docs and needle in self.norm[name]]
            if clashes:
                # 不同规章制度共用"48 小时内上报"这类通用时限是常态，不必当 bug；
                # 真正的问题是**所有**要点都能在别人家里找到，那这题没法判卷。
                self.warnings.append(f"{where}：要点「{point}」在 {clashes} 里也逐字出现（判别力弱）")
            else:
                unique_points += 1
        if points and unique_points == 0:
            self.errors.append(f"{where}：{len(points)} 个要点没有一个只出现在期望文档里，无法据引用判卷")

        sections = [str(sec) for sec in (item.get("expected_sections") or []) if str(sec).strip()]
        if not sections:
            self.warnings.append(f"{where}：没写 expected_sections，章节级诊断只能靠猜")
        pool: set[str] = set()
        for doc in docs:
            pool |= self.articles[doc]
        pool_norm = {normalize(value) for value in pool}
        for section in sections:
            leaf = section.split(">")[-1].strip()
            if normalize(leaf) not in pool_norm:
                self.errors.append(
                    f"{where}：章节「{section}」不是 {docs} 里的条标题（可用：{sorted(pool)[:3]}…）"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 data/bench 合成语料与题集")
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--raw", type=Path, default=None, help="语料目录（默认 <bench>/raw）")
    parser.add_argument("--golden", action="append", default=[], help="指定题集文件，可重复（默认 golden_*.jsonl）")
    parser.add_argument("--allow-unlabeled", action="store_true", help="暂时容忍缺失 variant/domain")
    parser.add_argument(
        "--min-doc-coverage",
        type=float,
        default=0.85,
        help="至少多少比例的文档要被题目引用，低于此值判为题集未写完",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    # Windows 控制台默认 cp936，中文报错会糊成乱码；只在现编码不是 UTF-8 时改（pytest 捕获流已经是 UTF-8）
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if "utf" not in encoding and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    raw_dir = args.raw or (args.bench_dir / "raw")
    if not raw_dir.is_dir():
        print(f"语料目录不存在：{raw_dir}", file=sys.stderr)
        return 2
    files = [Path(f) for f in args.golden] or sorted(args.bench_dir.glob("golden_*.jsonl"))
    if not files:
        print(f"没有找到题集：{args.bench_dir}/golden_*.jsonl", file=sys.stderr)
        return 2

    checker = Checker(raw_dir)
    checker.load_docs()
    checker.check_goldens(files, allow_unlabeled=args.allow_unlabeled, min_doc_coverage=args.min_doc_coverage)

    if args.json:
        print(json.dumps({"stats": checker.stats, "errors": checker.errors, "warnings": checker.warnings},
                         ensure_ascii=False, indent=2))
    else:
        for message in checker.errors:
            print(f"ERROR  {message}")
        for message in checker.warnings:
            print(f"WARN   {message}")
        print(
            f"\n语料 {checker.stats['docs']} 篇｜被引用 {checker.stats['docs_referenced']} 篇"
            f"（{checker.stats['doc_coverage']:.0%}）｜题集 {len(files)} 份 {checker.stats['items']} 题｜"
            f"variant {checker.stats['by_variant']}｜error {len(checker.errors)}｜warning {len(checker.warnings)}"
        )
    return 1 if checker.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
