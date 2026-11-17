r"""公开中文检索集对照：在 T2Ranking dev 子集上跑本项目同一套检索组件。

为什么要这一张表：`data/bench/` 那 48 篇是**合成**语料（`data/bench/SPEC.md` 写明了），
`scripts/ablation.py` 在上面拿到的候选 Hit@1=100% 只能证明"链路没坏 + 干扰项设计生效"，
外面没人认这个数。T2Ranking（SIGIR 2023，Apache-2.0）有真人标的 4 级相关性，
把它接上之后，"BM25 / 向量 / RRF / 精排各值多少分"第一次有第三方可以对表的口径。

数据来源与拼接（由 ``scripts/fetch_t2ranking.py`` 准备好，本脚本不联网）：

- 语料 + 问题：``C-MTEB/T2Retrieval`` 的 parquet（118,605 段 / 22,812 条问题）。
  **它自己不带标注**，所以只下载它是没法判卷的。
- 标注：``THUIR/T2Ranking`` 的 ``qrels.retrieval.dev.tsv``（二值 qid,pid，判 Hit@k / MRR）
  与 ``qrels.dev.tsv``（四级 qid - pid rel，判 nDCG@10）。id 与 parquet 同一套编号，
  fetch 脚本会核对"正例是否全部落在语料池里"，不通过就退出码非 0。

口径要说清楚的三件事，否则这张表会骗人：

1. **语料池只有官方 collection 的 5%**（118,605 / 2,303,643）。任务变简单了，
   绝对分数不能和官方 dev 基线（BM25 MRR@10 0.359、DPR+CE 0.551）直接比；
   能比的是**我们这几行之间的相对关系**，以及"融合/精排是否仍然加分"这个结论本身。
2. **只测检索**：不走 chroma、不走分块器、不做拒答闸门和逐句校验。它回答
   "检索组件本身值多少分"，和端到端的 `ablation.py` 是两个问题，别互相解释。
3. **同一批问题跑所有行**：精排那行最贵，所以行数由 ``--limit`` 控制，
   四行用的是完全相同的抽样（固定 ``--seed``），不存在"某一行多做了几道题"。

用法::

    python scripts/fetch_t2ranking.py                       # 先取数（约 160 MB）
    python scripts/external_bench.py --limit 50             # 冒烟，只验证脚本跑得通
    python scripts/external_bench.py --limit 1000           # 主结果
    python scripts/external_bench.py --limit 22812          # 全量 dev（慢，约数小时）

``RAG_FORCE_MOCK=1`` 会把 embedding 退化成哈希向量，那时数字只能证明链路通，
脚本会显式警告"不可用于对比"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_assistant.config import Settings, get_settings  # noqa: E402
from rag_assistant.embeddings import Embedder, get_embedder  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402
from rag_assistant.retrieval.bm25 import BM25Index, BM25Record, ChineseTokenizer  # noqa: E402
from rag_assistant.retrieval.fusion import rrf_fuse  # noqa: E402
from rag_assistant.retrieval.reranker import Reranker, get_reranker  # noqa: E402

DEFAULT_DIR = PROJECT_ROOT / "data" / "external" / "t2ranking"

_TAG_RE = re.compile(r"<[^>]{0,40}>")
_WS_RE = re.compile(r"\s+")

# 官方 README 公布的 dev 集基线（全量 2,303,643 段语料池），照抄用于说明量级方向。
# 来源：https://github.com/THUIR/T2Ranking （"Data Download" 下方的结果段）
OFFICIAL_BASELINES = (
    ("BM25", "0.3589", "0.0510", "0.4943", "-"),
    ("DPR（BM25 难负例）", "0.4856", "0.0737", "0.7099", "-"),
    ("DPR（自挖难负例）", "0.5167", "0.0805", "0.7327", "-"),
    ("BM25 召回 + 交叉编码器重排", "0.5188", "0.0855", "0.5953", "0.4405"),
    ("DPR 召回 + 交叉编码器重排", "0.5509", "0.0890", "0.7394", "0.5131"),
)


def clean_text(text: str) -> str:
    """去掉 collection 里残留的 ``<br>``/``<img>`` 标记并压平空白。

    两边（BM25 与向量）用同一份清洗结果，否则某一行的分数会因为"标签被当成词"而虚高。
    """
    without_tags = _TAG_RE.sub(" ", text or "")
    return _WS_RE.sub(" ", without_tags).strip()


def load_qrels_binary(path: Path) -> dict[str, set[str]]:
    """``qid<TAB>pid``（首行表头）→ 每个问题的二值正例集合。"""
    out: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[0] == "qid":
                continue
            out.setdefault(parts[0], set()).add(parts[1])
    return out


def load_qrels_graded(path: Path) -> dict[str, dict[str, int]]:
    """TREC 格式 ``qid - pid rel``（首行表头）→ 四级相关性。"""
    out: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[0] == "qid":
                continue
            out.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return out


def load_parquet_table(path: Path, *, progress: bool = False) -> dict[str, str]:
    """parquet 的 ``id``/``text`` 两列 → ``{id: text}``。id 统一成 str 以便和 qrels 对齐。"""
    import pandas as pd

    if progress:
        print(f"[load] {path.name} ...", flush=True)
    frame = pd.read_parquet(path, columns=["id", "text"])
    return {str(pid): text for pid, text in zip(frame["id"], frame["text"], strict=True)}


@dataclass(frozen=True)
class Case:
    """一道有标注的检索题。``graded`` 可能比 ``positives`` 大（四级标注含部分相关）。"""

    qid: str
    query: str
    positives: frozenset[str]
    graded: Mapping[str, int] = field(default_factory=dict)


def build_cases(
    queries: Mapping[str, str],
    corpus_ids: set[str],
    binary: Mapping[str, set[str]],
    graded: Mapping[str, Mapping[str, int]],
    *,
    limit: int | None = None,
    seed: int = 20261117,
) -> list[Case]:
    """挑出"正例至少有一个落在语料池里"的问题，再按固定种子抽样。

    先过滤后抽样：直接对全量 dev 抽样会得到一批没有任何可召回正例的题，
    那批题把 Hit@1 平均分母撑大、分子撑不起，分数看着像崩了，其实是抽样口径错了。
    """
    usable = sorted(
        qid
        for qid, positives in binary.items()
        if qid in queries and (positives & corpus_ids)
    )
    if not usable:
        raise SystemExit("没有任何问题带可在语料池里命中的正例——先跑 scripts/fetch_t2ranking.py")
    if limit is not None and 0 < limit < len(usable):
        picked = sorted(random.Random(seed).sample(usable, limit))
    else:
        picked = usable
    cases: list[Case] = []
    for qid in picked:
        keep = binary[qid] & corpus_ids
        # 四级标注只保留"池子里存在且 rel>0"的段落：池外文档不可能被召回，
        # 把它算进 IDCG 分母等于用别人没给的语料扣分；rel=0 是官方标过的不相关，
        # 留在增益表里会让 nDCG 变成"奖励排得靠后"。
        graded_map = {
            pid: rel for pid, rel in (graded.get(qid) or {}).items() if pid in corpus_ids and rel > 0
        }
        cases.append(
            Case(
                qid=qid,
                query=clean_text(queries[qid]),
                positives=frozenset(keep),
                graded=graded_map,
            )
        )
    return cases


def select_corpus(
    corpus: Sequence[tuple[str, str]],
    cases: Sequence[Case],
    *,
    size: int | None = None,
    seed: int = 20261117,
) -> list[tuple[str, str]]:
    """语料池：默认全用；``size`` 指定时先保全部正例，再用随机段落补到该规模当干扰项。

    必须优先保正例——随机抽 10k 段落大概率把标注的正例抽没了，那样 Hit@1 恒为 0。
    """
    if not size or size >= len(corpus):
        return list(corpus)
    needed = {pid for case in cases for pid in case.positives}
    picked = [(pid, text) for pid, text in corpus if pid in needed]
    rest = [(pid, text) for pid, text in corpus if pid not in needed]
    room = max(0, size - len(picked))
    if room < len(rest):
        rest = random.Random(seed).sample(rest, room)
    picked.extend(rest)
    picked.sort(key=lambda item: item[0])
    return picked


# ---- 指标 ----


def hit_at_k(ranked: Sequence[str], positives: frozenset[str], k: int) -> int:
    return int(any(pid in positives for pid in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], positives: frozenset[str], k: int = 10) -> float:
    for position, pid in enumerate(ranked[:k], 1):
        if pid in positives:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked: Sequence[str], positives: frozenset[str], k: int) -> float:
    if not positives:
        return 0.0
    return len(positives & set(ranked[:k])) / len(positives)


def ndcg_at_k(ranked: Sequence[str], graded: Mapping[str, int], k: int = 10) -> float:
    """四级相关性下的 nDCG@k：增益 ``2**rel - 1``，折损 ``1/log2(rank+1)``。

    分母用「标注里最好的 k 条」排出来的理想 DCG；标注不足 k 条时按实际条数算，
    这样不会因为上游少标了文档就压低分数。
    """
    if not graded:
        return 0.0
    gains = [graded.get(pid, 0) for pid in ranked[:k]]
    dcg = sum((2**rel - 1) / math.log2(position + 1) for position, rel in enumerate(gains, 1) if rel > 0)
    ideal = sorted(graded.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(position + 1) for position, rel in enumerate(ideal, 1) if rel > 0)
    return dcg / idcg if idcg > 0 else 0.0


def _graded_for_ndcg(case: Case) -> dict[str, int]:
    """有四级标注就用四级，只有二值标注的题退化成 rel=1（MTEB 的判法）。"""
    return dict(case.graded) or {pid: 1 for pid in case.positives}


def compute_metrics(cases: Sequence[Case], rankings: Mapping[str, list[str]], *, ks=(1, 10, 50, 100)) -> dict[str, Any]:
    """在**候选池内**判卷：正例取 ``case.positives``（已过滤成池内存在的 pid）。"""
    hits = {k: [] for k in ks}
    mrr, ndcg, recall = [], [], []
    for case in cases:
        ranked = rankings.get(case.qid, [])
        for k in ks:
            hits[k].append(hit_at_k(ranked, case.positives, k))
        mrr.append(reciprocal_rank(ranked, case.positives, 10))
        ndcg.append(ndcg_at_k(ranked, _graded_for_ndcg(case), 10))
        recall.append(recall_at_k(ranked, case.positives, 10))
    out: dict[str, Any] = {"n_queries": len(cases)}
    for k in ks:
        out[f"hit@{k}"] = sum(hits[k]) / len(hits[k]) if hits[k] else 0.0
    count = max(1, len(cases))
    out["mrr@10"] = sum(mrr) / count
    out["ndcg@10"] = sum(ndcg) / count
    out["recall@10"] = sum(recall) / count
    return out


# ---- 检索侧 ----


def load_or_encode_corpus(
    embedder: Embedder,
    pids: Sequence[str],
    texts: Sequence[str],
    *,
    cache_dir: Path,
    batch: int = 512,
    no_cache: bool = False,
) -> np.ndarray:
    """语料向量缓存：文件名带模型名 + 维度 + 池指纹，池一变就必然 miss。

    11.8 万段的编码要十几分钟，跑错一个参数就重来是不可接受的；但缓存也不能
    按"大小"命名，否则换 seed 后同一份名字对应不同段落，静默用错向量比慢更糟。
    """
    fingerprint = hashlib.md5("\n".join(pids).encode("utf-8")).hexdigest()[:10]
    model = re.sub(r"[^0-9A-Za-z._-]+", "_", embedder.__class__.__name__ + "_" + str(getattr(embedder, "dim", 0)))
    cache_path = cache_dir / f"vec_{model}_{len(pids)}_{fingerprint}.npy"
    if not no_cache and cache_path.exists():
        matrix = np.load(cache_path)
        if matrix.shape == (len(pids), embedder.dim):
            print(f"[cache] 复用 {cache_path.name}（{matrix.shape[0]:,} × {matrix.shape[1]}）", flush=True)
            return matrix
        print(f"[cache] {cache_path.name} 形状不符（{matrix.shape}），重新编码", flush=True)
    started = time.perf_counter()
    matrix = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    for start in range(0, len(texts), batch):
        piece = list(texts[start : start + batch])
        matrix[start : start + len(piece)] = np.asarray(embedder.embed_documents(piece), dtype=np.float32)
        done = start + len(piece)
        if done % (batch * 10) == 0 or done == len(texts):
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (len(texts) - done) / rate if rate > 0 else 0.0
            print(f"       编码 {done:,}/{len(texts):,}  {rate:.0f} 段/秒  剩余 {eta/60:.1f} 分钟", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    print(f"[cache] 已写入 {cache_path.name}", flush=True)
    return matrix


def top_k_vectors(passage_matrix: np.ndarray, query_matrix: np.ndarray, k: int) -> list[list[int]]:
    """暴力内积取 top-k（向量已归一，内积即余弦）。有 CUDA 就用，否则 numpy 分批。"""
    if passage_matrix.shape[0] == 0 or query_matrix.shape[0] == 0:
        return [[] for _ in range(query_matrix.shape[0])]
    top_k = min(k, passage_matrix.shape[0])
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            passages = torch.from_numpy(passage_matrix).to(device)
            out: list[list[int]] = []
            for start in range(0, query_matrix.shape[0], 128):
                batch = torch.from_numpy(query_matrix[start : start + 128]).to(device)
                sims = batch @ passages.T
                indices = torch.topk(sims, top_k, dim=1).indices
                out.extend(row.tolist() for row in indices.cpu())
            return out
    except ImportError:  # pragma: no cover - 没装 torch 才走 numpy
        pass
    out = []
    for row in query_matrix:
        sims = passage_matrix @ row
        indices = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k] if top_k < len(sims) else np.arange(len(sims))
        ordered = indices[np.argsort(-sims[indices], kind="stable")]
        out.append([int(index) for index in ordered])
    return out


def build_bm25_index(corpus: Sequence[tuple[str, str]], settings: Settings) -> BM25Index:
    """用线上那套分词（自定义词典 + 停用词 + BM25Plus）建索引，不另写一个简化版。"""
    tokenizer = ChineseTokenizer(
        userdict_path=PROJECT_ROOT / settings.retrieval.userdict_path,
        stopwords_path=PROJECT_ROOT / settings.retrieval.stopwords_path,
    )
    index = BM25Index(tokenizer)
    index.add([BM25Record(chunk_id=pid, text=text, metadata={"doc_id": pid}) for pid, text in corpus])
    return index


def rerank_ids(
    reranker: Reranker,
    query: str,
    candidate_ids: Sequence[str],
    text_of: Mapping[str, str],
) -> list[str]:
    """按精排分数降序返回候选 id；分数并列时保持 RRF 原序（``sorted`` 稳定）。"""
    if not candidate_ids:
        return []
    scores = reranker.score(query, [text_of[pid] for pid in candidate_ids])
    return [pid for pid, _ in sorted(zip(candidate_ids, scores, strict=True), key=lambda item: -item[1])]


@dataclass(frozen=True)
class RowSpec:
    name: str
    purpose: str
    key: str


ROWS = (
    RowSpec("BM25 only", "纯词面召回的地板", "bm25"),
    RowSpec("向量 only", "纯语义召回的地板", "dense"),
    RowSpec("+RRF", "两路融合、不重排", "rrf"),
    RowSpec("+rerank", "融合候选送交叉编码器精排", "rerank"),
)


def run_rows(
    cases: Sequence[Case],
    corpus: Sequence[tuple[str, str]],
    *,
    embedder: Embedder,
    reranker: Reranker | None,
    settings: Settings,
    cache_dir: Path,
    candidate_k: int,
    rerank_candidates: int,
    rrf_k: int,
    encode_batch: int,
    no_cache: bool,
    only: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """四行共用同一份召回结果：先算 bm25/dense 各 ``candidate_k`` 条，再融合、再精排。

    复用而不是每行重跑，是为了保证"+RRF 比两路好"这句话里两路的输入完全相同；
    否则分数差异可能来自某一行多召回了几条。
    """
    pids = [pid for pid, _ in corpus]
    texts = [text for _, text in corpus]
    text_of = dict(corpus)
    wanted = [row for row in ROWS if not only or row.key in only or row.name in only]

    print(f"[bm25] 建索引 {len(corpus):,} 段 ...", flush=True)
    index = build_bm25_index(corpus, settings)
    started = time.perf_counter()
    bm25_lists = [
        [hit.chunk_id for hit in index.search(case.query, candidate_k)] for case in cases
    ]
    bm25_ms = (time.perf_counter() - started) * 1000 / max(1, len(cases))
    print(f"       平均 {bm25_ms:.1f} ms/题", flush=True)

    need_dense = any(row.key != "bm25" for row in wanted)
    dense_lists: list[list[str]] = [[] for _ in cases]
    dense_ms = 0.0
    if need_dense:
        passage_matrix = load_or_encode_corpus(
            embedder, pids, texts, cache_dir=cache_dir, batch=encode_batch, no_cache=no_cache
        )
        started = time.perf_counter()
        query_matrix = np.asarray(embedder.embed_queries([case.query for case in cases]), dtype=np.float32)
        encode_query_ms = (time.perf_counter() - started) * 1000 / max(1, len(cases))
        started = time.perf_counter()
        dense_idx = top_k_vectors(passage_matrix, query_matrix, candidate_k)
        dense_ms = (time.perf_counter() - started) * 1000 / max(1, len(cases)) + encode_query_ms
        dense_lists = [[pids[index_] for index_ in row] for row in dense_idx]
        print(f"       向量检索平均 {dense_ms:.1f} ms/题", flush=True)

    results: list[dict[str, Any]] = []
    for row in wanted:
        started = time.perf_counter()
        if row.key == "bm25":
            rankings = {case.qid: bm25_lists[i] for i, case in enumerate(cases)}
            latency = bm25_ms
        elif row.key == "dense":
            rankings = {case.qid: dense_lists[i] for i, case in enumerate(cases)}
            latency = dense_ms
        else:
            fused: dict[str, list[str]] = {}
            for i, case in enumerate(cases):
                merged = rrf_fuse([bm25_lists[i], dense_lists[i]], k=rrf_k)
                fused[case.qid] = [pid for pid, _ in merged]
            if row.key == "rrf":
                rankings = fused
                latency = dense_ms
            else:
                if reranker is None:
                    raise SystemExit("该行需要 reranker，但配置里没启用：检查 reranker.enabled / RAG_FORCE_MOCK")
                reranked: dict[str, list[str]] = {}
                for case in cases:
                    head = fused[case.qid][:rerank_candidates]
                    reranked[case.qid] = rerank_ids(reranker, case.query, head, text_of)
                latency = (time.perf_counter() - started) * 1000 / max(1, len(cases))
                rankings = reranked
        results.append(
            {
                "key": row.key,
                "name": row.name,
                "purpose": row.purpose,
                "metrics": compute_metrics(cases, rankings),
                "latency_ms": latency,
                "n_cand_zero": sum(1 for case in cases if not rankings.get(case.qid)),
            }
        )
        metrics = results[-1]["metrics"]
        print(
            f"[row] {row.name}  Hit@1 {metrics['hit@1']*100:.2f}%  Hit@10 {metrics['hit@10']*100:.2f}%  "
            f"MRR@10 {metrics['mrr@10']:.4f}  nDCG@10 {metrics['ndcg@10']:.4f}",
            flush=True,
        )
    return results


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = [
        f"# 公开集对照 · T2Ranking（{meta['label']}）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 0. 运行条件",
        "",
    ]
    for key, value in meta.items():
        if key == "label":
            continue
        lines.append(f"- {key}：{value}")
    lines += [
        "",
        "**这张表只测检索**：语料是别人标的公开段落，没有本项目的分块、阈值和拒答闸门。",
        "它回答「检索组件本身值多少分」；端到端（含闸门、含答案引用）看 `scripts/ablation.py`。",
        "语料池是官方 collection 的一个子集，绝对值与官方基线不可直接比，见第 3 节。",
        "",
        "## 1. 我们这几行",
        "",
        "| 配置 | Hit@1 | Hit@10 | MRR@10 | nDCG@10 | Recall@10 | 零命中题 | 每题为 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| {row['name']} | {_pct(m['hit@1'])} | {_pct(m['hit@10'])} | {m['mrr@10']:.4f} "
            f"| {m['ndcg@10']:.4f} | {_pct(m['recall@10'])} | {row['n_cand_zero']} "
            f"| {row['latency_ms']:.1f}ms |"
        )
    lines += [
        "",
        "## 2. 每行实际怎么算的",
        "",
        "| 行 | 候选来源 | 送给精排 | RRF k |",
        "|---|---|---|---|",
    ]
    for row in rows:
        source = {"bm25": "BM25Plus（jieba + 本项目停用词表）", "dense": "bge-m3 余弦（归一向量暴力 top-k）"}.get(row["key"])
        if source is None:
            source = "BM25 top-N 与向量 top-N 的 RRF 融合"
        lines.append(
            f"| {row['name']} | {source} | {meta['精排候选数'] if row['key'] == 'rerank' else '—'} "
            f"| {meta['RRF_k']} |"
        )
    lines += [
        "",
        "## 3. 官方 dev 基线（全量 2,303,643 段语料池，来自 THUIR/T2Ranking README）",
        "",
        "| 系统 | MRR@10 | Recall@1 | Recall@50 | nDCG@20 |",
        "|---|---|---|---|---|",
    ]
    for name, mrr, r1, r50, ndcg20 in OFFICIAL_BASELINES:
        lines.append(f"| {name} | {mrr} | {r1} | {r50} | {ndcg20} |")
    pool = int(meta.get("语料池段数") or 0)
    official = 2_303_643
    if pool and pool >= official * 0.9:
        pool_note = (
            f"1. 本表语料池 {pool:,} 段，与官方基线的 {official:,} 段同规模，"
            "两栏可以直接比（分词器与截断长度的差别见第 3 点）。"
        )
    else:
        ratio = f"{official / pool:.0f}" if pool else "若干"
        pool_note = (
            f"1. 本表语料池 {pool:,} 段，官方基线是 {official:,} 段——本表只有它的 1/{ratio}。"
            "池子越小，随机命中概率越高，所以本表分数**系统性偏高**，不能拿去和上面的数字比大小。"
        )
    lines += [
        "",
        "对比时请注意三件事：",
        "",
        pool_note,
        "2. 官方 Recall@1 判的是「排第一的是不是正例」，与本表 Hit@1 同义；"
        "nDCG@20 与本表 nDCG@10 截断长度不同。",
        "3. 本表 BM25 用 BM25Plus + jieba（与线上检索器同一份代码），官方 BM25 是 pyserini + 中文分词器，"
        "词面这一路的地板高度不来自同一套参数。",
        "",
        "**能拿来下的结论只有相对关系**：哪一路是地板、融合值不值、精排值不值。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在公开中文检索集上对照检索组件")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="fetch_t2ranking.py 的落盘目录")
    parser.add_argument("--limit", type=int, default=1000, help="抽多少道题（0 = 全量）")
    parser.add_argument("--corpus-size", type=int, default=0, help="语料池规模（0 = 全用 parquet 里的 118,605 段）")
    parser.add_argument("--candidate-k", type=int, default=100, help="每路召回多少条（融合前的输入）")
    parser.add_argument("--rerank-candidates", type=int, default=50, help="精排输入条数")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--encode-batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--no-cache", action="store_true", help="忽略向量缓存，重新编码")
    parser.add_argument("--only", action="append", default=[], help="只跑指定行（key 或名字，可重复）")
    parser.add_argument("--label", default="t2ranking", help="报告标签")
    parser.add_argument("--out", type=Path, default=None, help="markdown 输出路径")
    parser.add_argument("--json", action="store_true", help="把每行指标打到 stdout")
    args = parser.parse_args(argv)

    setup_logging()
    data_dir: Path = args.dir if args.dir.is_absolute() else PROJECT_ROOT / args.dir
    missing = [
        name
        for name in ("corpus.parquet", "queries.parquet", "qrels.retrieval.dev.tsv", "qrels.dev.tsv")
        if not (data_dir / name).exists()
    ]
    if missing:
        print(f"{data_dir} 缺少 {missing}：先跑 python scripts/fetch_t2ranking.py", file=sys.stderr)
        return 2

    settings = get_settings()
    embedder = get_embedder(settings)
    reranker = get_reranker(settings.reranker)
    if embedder.is_mock:
        print(
            "警告：embedder 处于 mock 模式（哈希向量）。本次只能验证脚本，向量行与融合行的数字不可用于对比。",
            file=sys.stderr,
        )
    if reranker is None:
        print("提示：reranker 未启用，+rerank 行会被跳过。", file=sys.stderr)

    corpus_map = load_parquet_table(data_dir / "corpus.parquet", progress=True)
    query_map = load_parquet_table(data_dir / "queries.parquet", progress=True)
    binary = load_qrels_binary(data_dir / "qrels.retrieval.dev.tsv")
    graded = load_qrels_graded(data_dir / "qrels.dev.tsv")
    print(f"[load] 语料 {len(corpus_map):,} 段｜问题 {len(query_map):,} 条｜标注问题 {len(binary):,} 条", flush=True)

    cases = build_cases(
        query_map,
        set(corpus_map),
        binary,
        graded,
        limit=args.limit or None,
        seed=args.seed,
    )
    corpus = select_corpus(
        [(pid, clean_text(text)) for pid, text in sorted(corpus_map.items(), key=lambda item: str(item[0]))],
        cases,
        size=args.corpus_size or None,
        seed=args.seed,
    )
    print(f"[case] {len(cases)} 题，每题平均 {sum(len(c.positives) for c in cases)/max(1,len(cases)):.2f} 个池内正例｜语料池 {len(corpus):,} 段", flush=True)

    rows = run_rows(
        cases,
        corpus,
        embedder=embedder,
        reranker=reranker,
        settings=settings,
        cache_dir=data_dir,
        candidate_k=args.candidate_k,
        rerank_candidates=args.rerank_candidates,
        rrf_k=args.rrf_k,
        encode_batch=args.encode_batch,
        no_cache=args.no_cache,
        only=args.only,
    )

    meta = {
        "label": args.label,
        "题集": f"THUIR/T2Ranking dev（{len(cases)} 题，seed={args.seed}）",
        "语料池": f"{len(corpus):,} 段（来自 C-MTEB/T2Retrieval parquet，官方 collection 为 2,303,643 段）",
        "语料池段数": len(corpus),
        "标注": "qrels.retrieval.dev.tsv 二值判 Hit/MRR，qrels.dev.tsv 四级判 nDCG",
        "Embedding": f"{settings.embedding.model}（mock={embedder.is_mock}，device={getattr(embedder, 'device', '-')}）",
        "Reranker": f"{settings.reranker.model}（enabled={reranker is not None}）",
        "每路召回": args.candidate_k,
        "精排候选数": args.rerank_candidates,
        "RRF_k": args.rrf_k,
        "许可": "Apache-2.0（数据集与论文见 https://github.com/THUIR/T2Ranking）",
    }
    report = render(rows, meta)
    out_path = args.out or (PROJECT_ROOT / "reports" / f"external-{args.label}-{datetime.now().strftime('%Y%m%d-%H%M')}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入：{out_path}")
    if args.json:
        print(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
