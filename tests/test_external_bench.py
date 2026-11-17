"""公开集对照脚本（``scripts/external_bench.py``、``scripts/fetch_t2ranking.py``）的单测。

只盯会**静默把分数算错**的那几处：标注解析、先过滤后抽样、nDCG 分母、指标分母、
语料池保正例、向量缓存命名。真实模型与网络都不进来——``run_rows`` 用 HashEmbedder
+ MockReranker，语料是手写的六段，整套跑完不到一秒。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

import external_bench as eb
import fetch_t2ranking as fetch
from rag_assistant.embeddings.mock import HashEmbedder
from rag_assistant.retrieval.reranker import MockReranker

# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------


def test_clean_text_strips_markup_and_folds_whitespace() -> None:
    assert eb.clean_text("休学<br><img>申请 ，  一年内") == "休学 申请 ， 一年内"
    assert eb.clean_text("") == ""


def test_load_qrels_skips_header_and_malformed_lines(tmp_path: Path) -> None:
    binary = tmp_path / "qrels.retrieval.dev.tsv"
    binary.write_text("qid\tpid\n0\t159474\n0\t159475\n只有一列\n1\t7\n", encoding="utf-8")
    graded = tmp_path / "qrels.dev.tsv"
    graded.write_text("qid\t-\tpid\trel\n0\t0\t159474\t3\n1\t0\t7\t0\n\n", encoding="utf-8")

    assert eb.load_qrels_binary(binary) == {"0": {"159474", "159475"}, "1": {"7"}}
    assert eb.load_qrels_graded(graded) == {"0": {"159474": 3}, "1": {"7": 0}}


# --------------------------------------------------------------------------
# 抽样与语料池：这两步错了，Hit@1 会恒为 0 或恒为满分，而且看不出问题
# --------------------------------------------------------------------------

CORPUS = [("1", "学生因病休学 应当办理 休学手续"), ("2", "国家奖学金 评审 名额"), ("3", "差旅费 报销 十五个工作日"), ("4", "档案 转递 机要 通道")]
QUERIES = {"0": "休学 手续", "1": "奖学金 怎么 评", "2": "报销 时限", "3": "池外正例"}
BINARY = {"0": {"1"}, "1": {"2"}, "2": {"3"}, "3": {"999"}}
GRADED = {"0": {"1": 3, "4": 1, "999": 3}, "1": {"2": 2}, "2": {"3": 3}, "3": {"999": 3}}


def make_cases(**kwargs: object) -> list[eb.Case]:
    corpus_ids = {pid for pid, _ in CORPUS}
    return eb.build_cases(QUERIES, corpus_ids, BINARY, GRADED, **kwargs)  # type: ignore[arg-type]


def test_build_cases_drops_questions_whose_positives_are_out_of_pool() -> None:
    cases = make_cases()
    assert [case.qid for case in cases] == ["0", "1", "2"], "正例全在池外的题必须丢掉，否则 Hit@1 白扣"
    by_qid = {case.qid: case for case in cases}
    assert by_qid["0"].positives == frozenset({"1"})
    assert by_qid["0"].graded == {"1": 3, "4": 1}, "池外正例（999）不能进 IDCG 分母"


def test_build_cases_excludes_rel_zero_from_graded() -> None:
    graded = {**GRADED, "1": {"2": 2, "4": 0}}
    cases = eb.build_cases(QUERIES, {pid for pid, _ in CORPUS}, BINARY, graded)
    assert {case.qid: case.graded for case in cases}["1"] == {"2": 2}


def test_build_cases_raises_when_nothing_is_usable() -> None:
    with pytest.raises(SystemExit):
        eb.build_cases(QUERIES, {"不存在的 pid"}, BINARY, GRADED)


def test_build_cases_samples_after_filtering_and_is_deterministic() -> None:
    first = [case.qid for case in make_cases(limit=2, seed=3)]
    second = [case.qid for case in make_cases(limit=2, seed=3)]
    assert len(first) == 2
    assert first == second, "同一个 seed 必须复现同一批题，否则两次跑的表没法比"
    assert set(first) <= {"0", "1", "2"}, "抽样只能落在过滤后的可用题里"


def test_select_corpus_keeps_every_positive() -> None:
    cases = make_cases()
    needed = {pid for case in cases for pid in case.positives}
    picked = eb.select_corpus(CORPUS, cases, size=3, seed=1)

    assert needed <= {pid for pid, _ in picked}, "补干扰项不能把正例挤掉"
    assert len(picked) == 3
    assert [pid for pid, _ in picked] == sorted(pid for pid, _ in picked), "池子要按 id 定序，向量缓存指纹才稳定"


def test_select_corpus_without_size_or_oversized_returns_all() -> None:
    cases = make_cases()
    assert eb.select_corpus(CORPUS, cases, size=None) == CORPUS
    assert eb.select_corpus(CORPUS, cases, size=999) == CORPUS


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------


def test_hit_at_k_and_recall() -> None:
    positives = frozenset({"b", "z"})
    assert eb.hit_at_k(["a", "b", "c"], positives, 1) == 0
    assert eb.hit_at_k(["a", "b", "c"], positives, 2) == 1
    assert eb.recall_at_k(["a", "b", "c"], positives, 2) == pytest.approx(1 / 2)
    assert eb.recall_at_k(["a"], frozenset(), 1) == 0.0


def test_reciprocal_rank_only_counts_first_hit_within_cut() -> None:
    assert eb.reciprocal_rank(["x", "y", "b"], frozenset({"b"}), 10) == pytest.approx(1 / 3)
    assert eb.reciprocal_rank(["x", "y", "b"], frozenset({"b"}), 2) == 0.0


def test_ndcg_at_k_hand_computed() -> None:
    graded = {"a": 3, "b": 2}
    assert eb.ndcg_at_k(["a", "b"], graded, 2) == 1.0, "理想排序必须是 1"
    # 增益 2**rel-1，折损 1/log2(名次+1)：a 在第 1 位、b 在第 3 位，中间插一条不相关
    expected = (7 / math.log2(2) + 3 / math.log2(4)) / (7 / math.log2(2) + 3 / math.log2(3))
    assert eb.ndcg_at_k(["a", "x", "b"], graded, 3) == pytest.approx(expected)
    assert eb.ndcg_at_k(["x", "y"], {}, 10) == 0.0
    assert eb.ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 2) == 0.0, "全 rel=0 时不能除零"


def test_ndcg_falls_back_to_binary_when_no_graded_labels() -> None:
    case = eb.Case(qid="0", query="q", positives=frozenset({"a", "b"}))
    graded = eb._graded_for_ndcg(case)
    assert graded == {"a": 1, "b": 1}
    assert eb.ndcg_at_k(["a", "b"], graded, 10) == 1.0


def test_compute_metrics_divides_by_question_count() -> None:
    """踩过的坑：分子按题累加、分母误用了 ks 的个数，Hit@1 打成 900%。"""
    cases = [eb.Case(qid=str(i), query="q", positives=frozenset({"p"})) for i in range(40)]
    metrics = eb.compute_metrics(cases, {case.qid: ["p", "x"] for case in cases}, ks=(1, 10))

    assert metrics["n_queries"] == 40
    assert metrics["hit@1"] == 1.0
    assert metrics["hit@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] == 1.0
    assert metrics["recall@10"] == 1.0


def test_compute_metrics_counts_missing_rankings_as_zero() -> None:
    cases = [eb.Case(qid=str(i), query="q", positives=frozenset({"p"})) for i in range(4)]
    metrics = eb.compute_metrics(cases, {"0": ["p"], "1": ["p"]}, ks=(1, 10))
    assert metrics["hit@1"] == 0.5
    assert metrics["mrr@10"] == 0.5


# --------------------------------------------------------------------------
# 检索侧
# --------------------------------------------------------------------------


def test_top_k_vectors_matches_brute_force() -> None:
    passages = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0], [0, 0, 1]], dtype=np.float32)
    query = np.array([[0.9, 0.8, 0.1], [0.0, 0.1, 1.0]], dtype=np.float32)
    ranked = eb.top_k_vectors(passages, query, 2)

    assert len(ranked) == query.shape[0]
    for position, row in enumerate(ranked):
        sims = passages @ query[position]
        assert row == [int(i) for i in np.argsort(-sims)[:2]]
    assert eb.top_k_vectors(np.zeros((0, 3), dtype=np.float32), query, 5) == [[], []]


def test_rerank_ids_sorts_by_score_and_keeps_input_order_on_ties() -> None:
    class ConstantScore:
        name = "constant"
        is_mock = False

        def score(self, query: str, texts: Sequence[str]) -> list[float]:  # noqa: ARG002
            return [0.1, 0.9, 0.1, 0.1]

    ordered = eb.rerank_ids(ConstantScore(), "q", list("abcd"), {pid: pid for pid in "abcd"})
    assert ordered == ["b", "a", "c", "d"], "同分要保持 RRF 原序（sorted 是稳定的）"
    assert eb.rerank_ids(ConstantScore(), "q", [], {}) == []


def _tiny_setup() -> tuple[list[eb.Case], list[tuple[str, str]]]:
    cases = make_cases()
    return cases, CORPUS


def test_run_rows_returns_all_four_rows_sharing_the_same_candidates(settings, tmp_path: Path) -> None:
    cases, corpus = _tiny_setup()
    rows = eb.run_rows(
        cases,
        corpus,
        embedder=HashEmbedder(dim=64),
        reranker=MockReranker(),
        settings=settings,
        cache_dir=tmp_path,
        candidate_k=3,
        rerank_candidates=2,
        rrf_k=60,
        encode_batch=8,
        no_cache=True,
    )

    assert [row["key"] for row in rows] == ["bm25", "dense", "rrf", "rerank"]
    assert rows[0]["metrics"]["hit@10"] == 1.0, "词面这一路的正例就在候选里，命中不了说明拼接错了"
    for row in rows:
        assert row["metrics"]["n_queries"] == len(cases)
        assert row["n_cand_zero"] == 0
        assert row["latency_ms"] >= 0


def test_run_rows_only_filter_accepts_key_and_name(settings, tmp_path: Path) -> None:
    cases, corpus = _tiny_setup()
    kwargs = dict(
        embedder=HashEmbedder(dim=64),
        reranker=MockReranker(),
        settings=settings,
        cache_dir=tmp_path,
        candidate_k=3,
        rerank_candidates=2,
        rrf_k=60,
        encode_batch=8,
        no_cache=True,
    )
    by_key = eb.run_rows(cases, corpus, only=["bm25"], **kwargs)  # type: ignore[arg-type]
    assert [row["key"] for row in by_key] == ["bm25"]

    by_name = eb.run_rows(cases, corpus, only=["+RRF"], **kwargs)  # type: ignore[arg-type]
    assert [row["key"] for row in by_name] == ["rrf"]


def test_run_rows_refuses_rerank_row_without_reranker(settings, tmp_path: Path) -> None:
    cases, corpus = _tiny_setup()
    with pytest.raises(SystemExit):
        eb.run_rows(
            cases,
            corpus,
            embedder=HashEmbedder(dim=64),
            reranker=None,
            settings=settings,
            cache_dir=tmp_path,
            candidate_k=3,
            rerank_candidates=2,
            rrf_k=60,
            encode_batch=8,
            no_cache=True,
            only=["rerank"],
        )


# --------------------------------------------------------------------------
# 向量缓存：跑错一个参数就重来十几分钟，但更糟的是静默用错向量
# --------------------------------------------------------------------------


class CountingEmbedder(HashEmbedder):
    """数一数真的编码了几批，用来证明"第二次是读缓存"。"""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_documents(texts)


def test_load_or_encode_corpus_reuses_cache_until_the_pool_changes(tmp_path: Path) -> None:
    pids = [pid for pid, _ in CORPUS]
    texts = [text for _, text in CORPUS]
    embedder = CountingEmbedder(dim=32)

    first = eb.load_or_encode_corpus(embedder, pids, texts, cache_dir=tmp_path, batch=2)
    calls_after_encode = embedder.calls
    again = eb.load_or_encode_corpus(embedder, pids, texts, cache_dir=tmp_path, batch=2)

    assert first.shape == (len(pids), 32)
    assert np.array_equal(first, again)
    assert embedder.calls == calls_after_encode, "同一份池子不该重编码"
    assert len(list(tmp_path.glob("vec_*.npy"))) == 1


def test_corpus_cache_key_follows_pool_membership(tmp_path: Path) -> None:
    texts = [text for _, text in CORPUS]
    embedder = CountingEmbedder(dim=32)
    eb.load_or_encode_corpus(embedder, [pid for pid, _ in CORPUS], texts, cache_dir=tmp_path)
    eb.load_or_encode_corpus(embedder, [pid for pid, _ in CORPUS[:3]], texts[:3], cache_dir=tmp_path)

    assert len(list(tmp_path.glob("vec_*.npy"))) == 2, "换池子必须换文件名，否则 seed 一变就静默用错向量"


def test_load_or_encode_corpus_reencodes_on_shape_mismatch(tmp_path: Path) -> None:
    pids = [pid for pid, _ in CORPUS]
    texts = [text for _, text in CORPUS]
    embedder = CountingEmbedder(dim=32)
    matrix = eb.load_or_encode_corpus(embedder, pids, texts, cache_dir=tmp_path)
    cache = next(tmp_path.glob("vec_*.npy"))

    np.save(cache, np.zeros((len(pids), 8), dtype=np.float32))
    calls_before = embedder.calls
    rebuilt = eb.load_or_encode_corpus(embedder, pids, texts, cache_dir=tmp_path)

    assert rebuilt.shape == matrix.shape
    assert embedder.calls > calls_before, "形状不符必须重编码，不能拿半截缓存算分"

    calls_before = embedder.calls
    eb.load_or_encode_corpus(embedder, pids, texts, cache_dir=tmp_path, no_cache=True)
    assert embedder.calls > calls_before, "no_cache 必须真的重编码"


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def test_render_states_provenance_and_the_comparison_caveats() -> None:
    cases, _ = _tiny_setup()
    rankings = {case.qid: [next(pid for pid, _ in CORPUS if pid in case.positives), "9"] for case in cases}
    rows = [
        {
            "key": row.key,
            "name": row.name,
            "purpose": row.purpose,
            "metrics": eb.compute_metrics(cases, rankings, ks=(1, 10)),
            "latency_ms": 1.0,
            "n_cand_zero": 0,
        }
        for row in eb.ROWS
    ]
    text = eb.render(
        rows,
        {"label": "t2ranking", "语料池": "6 段", "精排候选数": 2, "RRF_k": 60},
    )

    assert "## 1. 我们这几行" in text and "## 3. 官方 dev 基线" in text
    assert "只测检索" in text, "必须写清这张表不含分块/阈值/拒答闸门"
    assert "系统性偏高" in text, "池子只有官方的 5%，不写这句就会被拿去比大小"
    assert "100.00%" in text
    assert text.count("| +rerank |") == 2, "结果表和口径表都该有这一行"


# --------------------------------------------------------------------------
# 下载器：挑错文件 = 拿另一个划分判卷，分数看着正常但毫无意义
# --------------------------------------------------------------------------


def test_pick_file_requires_exactly_one_match() -> None:
    files = [
        {"path": "dev/queries.dev", "size": 10},
        {"path": "dev/corpus.parquet", "size": 20},
        {"path": "dev/corpus.parquet.metadata", "size": 1},
        {"path": "test/queries.test", "size": 30},
    ]
    assert fetch.pick_file(files, "dev/queries.dev") == files[0]
    assert fetch.pick_file(files, "corpus.parquet") == files[1], ".metadata 侧车文件不算命中"

    with pytest.raises(RuntimeError):
        fetch.pick_file(files, "queries")  # dev 与 test 各一个，不能猜
    with pytest.raises(RuntimeError):
        fetch.pick_file(files, "不存在的文件")


def test_download_skips_a_file_that_is_already_complete(tmp_path: Path) -> None:
    dest = tmp_path / "corpus.parquet"
    dest.write_bytes(b"12345")

    assert fetch.download("https://example.invalid/corpus.parquet", dest, expected_size=5) == dest
    assert dest.read_bytes() == b"12345", "大小对得上就不该碰原文件（更不该联网）"


def test_tree_files_rejects_an_error_payload() -> None:
    """API 报错时返回的是 dict，不能当成空目录混过去。"""

    def fake_get(url: str, timeout: int = 60) -> bytes:  # noqa: ARG001
        return b'{"error": 401, "message": "Repo not found"}'

    original = fetch._get
    fetch._get = fake_get
    try:
        with pytest.raises(RuntimeError):
            fetch.tree_files("a/b", "dev")
    finally:
        fetch._get = original
