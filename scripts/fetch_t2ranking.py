r"""下载公开中文检索集 THUIR/T2Ranking 的 dev 子集（语料 + 问题 + 相关性标注）。

为什么是这套：`data/bench/raw` 的 48 篇是**合成**语料（见 `data/bench/SPEC.md`），
它的 Hit@1=100% 只能证明链路没坏，跟任何外部数字都不可比。T2Ranking（SIGIR 2023，
Apache-2.0）是真人标注的中文段落排序基准，官方 README 还公布了 BM25 / DPR /
cross-encoder 的基线分数——只有把同一套组件放上去，我们的检索数字才第一次能被外人核对。

下载来源（两套都要，缺一不可）：

- `C-MTEB/T2Retrieval`：MTEB 自动转换出来的 parquet，`corpus` 118,605 段、
  `queries` 22,812 条。**只有文本，没有标注**——这是最常见的坑：直接拿它跑分
  会得到一个无法判卷的评测。
- `THUIR/T2Ranking`：官方 TSV 标注，取 `data/qrels.retrieval.dev.tsv`
  （qid, pid 二值，判 Hit@k / MRR）和 `data/qrels.dev.tsv`（qid - pid rel 四级，判
  nDCG@10）。两边的 id 是同一套编号，已核对 118,932 个正例 100% 落在 parquet 语料里。

全量 `collection.tsv` 是 3.6 GB，本脚本刻意不碰它：parquet 那份子集就是 dev 集中
"正例在库内"的部分，够大也够判卷，代价是语料池只有官方的 5%，报数字时必须写清楚。

用法::

    python scripts/fetch_t2ranking.py                 # 下到 data/external/t2ranking/
    python scripts/fetch_t2ranking.py --dir 某目录 --check-only
    HF_ENDPOINT=https://huggingface.co python scripts/fetch_t2ranking.py   # 梯子可用时

已有的同名同大小文件会跳过，所以中途断了可以接着跑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT_ROOT / "data" / "external" / "t2ranking"

# (repo, 子目录, 落盘文件名, 路径关键字) —— 靠关键字在 tree 里找，因为自动转换出来的
# parquet 文件名里带内容哈希，写死会在上游重新分片时失效。
PARQUET_SOURCES = [
    ("C-MTEB/T2Retrieval", "data", "corpus.parquet", "corpus"),
    ("C-MTEB/T2Retrieval", "data", "queries.parquet", "queries"),
]
TSV_SOURCES = [
    ("THUIR/T2Ranking", "data", "qrels.retrieval.dev.tsv", "data/qrels.retrieval.dev.tsv"),
    ("THUIR/T2Ranking", "data", "qrels.dev.tsv", "data/qrels.dev.tsv"),
]


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-assistant-bench/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 只走 https 固定域名
        return resp.read()


def tree_files(repo: str, subdir: str) -> list[dict[str, Any]]:
    """列仓库子目录下的文件（API 只用于列目录，下载走 resolve/main，不限流）。"""
    url = f"{endpoint()}/api/datasets/{repo}/tree/main/{subdir}"
    payload = json.loads(_get(url).decode("utf-8"))
    if isinstance(payload, dict):
        raise RuntimeError(f"{repo} 目录列举失败：{payload.get('message') or payload}")
    return [item for item in payload if item.get("type") == "file"]


def pick_file(files: list[dict[str, Any]], needle: str) -> dict[str, Any]:
    """按路径关键字挑文件；`needle` 是完整路径时精确匹配，否则匹配文件名前缀。

    必须只命中一个：命中两个会把 `queries.test` 当成 `queries.dev` 用，
    那是拿另一个划分判卷，分数看着正常但毫无意义。
    """
    hits = [item for item in files if needle in str(item.get("path", ""))]
    hits = [item for item in hits if not item["path"].endswith((".metadata", ".gitattributes"))]
    if not hits:
        raise RuntimeError(f"仓库里没找到 {needle}，可选：{[i['path'] for i in files][:12]}")
    if len(hits) > 1:
        raise RuntimeError(f"{needle} 命中多个文件，无法判断要哪个：{[i['path'] for i in hits]}")
    return hits[0]


def download(url: str, dest: Path, *, expected_size: int | None = None, chunk: int = 4 << 20) -> Path:
    """下载到 `.part` 再改名，避免半截文件被当成好的。已存在且大小对得上就跳过。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"[skip] {dest.name} 已存在（{dest.stat().st_size:,} 字节）")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "rag-assistant-bench/1.0"}
    if done:
        headers["Range"] = f"bytes={done}-"
        print(f"[resume] {dest.name} 从 {done:,} 字节继续")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        mode = "ab" if resp.status == 206 else "wb"
        if mode == "wb":
            done = 0
        total = resp.headers.get("Content-Length")
        total_bytes = (int(total) + done) if total and done else (int(total) if total else expected_size)
        with tmp.open(mode) as fh:
            last_report = done
            while True:
                blob = resp.read(chunk)
                if not blob:
                    break
                fh.write(blob)
                done += len(blob)
                if done - last_report >= (32 << 20):
                    pct = f"{done / total_bytes * 100:.1f}%" if total_bytes else "?"
                    print(f"       {dest.name}: {done:,}/{total_bytes or '?'} ({pct})", flush=True)
                    last_report = done
    tmp.replace(dest)
    if expected_size is not None and dest.stat().st_size != expected_size:
        raise RuntimeError(f"{dest.name} 大小不符：期望 {expected_size}，实际 {dest.stat().st_size}")
    print(f"[ok] {dest.name} {dest.stat().st_size:,} 字节")
    return dest


def verify(dir_path: Path) -> dict[str, Any]:
    """核对 id 对得上——这是整套拼接唯一的正确性证据，必须显式打印。"""
    import pandas as pd

    corpus = pd.read_parquet(dir_path / "corpus.parquet")
    queries = pd.read_parquet(dir_path / "queries.parquet")
    pids = {str(x) for x in corpus["id"]}
    qids = {str(x) for x in queries["id"]}

    binary: dict[str, set[str]] = {}
    with (dir_path / "qrels.retrieval.dev.tsv").open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] != "qid":
                binary.setdefault(parts[0], set()).add(parts[1])
    graded: dict[str, dict[str, int]] = {}
    with (dir_path / "qrels.dev.tsv").open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0] != "qid":
                graded.setdefault(parts[0], {})[parts[2]] = int(parts[3])

    scored = {qid: pos for qid, pos in binary.items() if qid in qids}
    pairs = sum(len(pos) for pos in scored.values())
    missing = sum(1 for pos in scored.values() for pid in pos if pid not in pids)
    text_match = sum(1 for _, row in queries.iterrows() if str(row["id"]) in graded)
    return {
        "corpus_passages": len(corpus),
        "parquet_queries": len(qids),
        "queries_with_positives": len(scored),
        "positive_pairs": pairs,
        "positives_missing_from_corpus": missing,
        "queries_with_graded_labels": text_match,
        "graded_pairs": sum(len(v) for v in graded.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 T2Ranking dev 子集（语料+标注）")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="落盘目录（默认 data/external/t2ranking）")
    parser.add_argument("--check-only", action="store_true", help="只校验已有文件，不下载")
    args = parser.parse_args(argv)

    target: Path = args.dir if args.dir.is_absolute() else PROJECT_ROOT / args.dir
    stats: dict[str, Any] = {}
    if not args.check_only:
        cache: dict[str, list[dict[str, Any]]] = {}
        for repo, subdir, name, needle in PARQUET_SOURCES + [
            (repo, subdir, name, needle) for repo, subdir, name, needle in TSV_SOURCES
        ]:
            key = f"{repo}:{subdir}"
            files = cache.setdefault(key, tree_files(repo, subdir))
            item = pick_file(files, needle)
            url = f"{endpoint()}/datasets/{repo}/resolve/main/{item['path']}"
            print(f"[get] {item['path']}  ({item.get('size', 0):,} 字节) ← {repo}")
            download(url, target / name, expected_size=int(item["size"]) if item.get("size") else None)
        stats["files"] = sorted(p.name for p in target.glob("*") if p.is_file())

    required = ["corpus.parquet", "queries.parquet", "qrels.retrieval.dev.tsv", "qrels.dev.tsv"]
    absent = [name for name in required if not (target / name).exists()]
    if absent:
        print(f"缺文件：{absent}（目录 {target}）", file=sys.stderr)
        return 2
    joined = verify(target)
    lines = [f"{key}：{value}" for key, value in joined.items()]
    lines.append(f"目录：{target}")
    out_path = target / "verify.json"
    out_path.write_text(json.dumps(joined, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    if joined["positives_missing_from_corpus"]:
        print("标注里的正例不在 parquet 语料里，判卷会漏——别用这套数据。", file=sys.stderr)
        return 1
    if joined["queries_with_positives"] < 1000:
        print("可用题数太少，跑出来的 Hit@1 抖动会很大。", file=sys.stderr)
        return 1
    print("校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
