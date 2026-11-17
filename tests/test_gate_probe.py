"""`scripts/gate_probe.py` 的测试：用假 service，只验"记下来的数对不对、重放会不会说谎"。

这个脚本的存在理由是一次真实的自欺：`coverage_relax_score=0.75` 在真实精排分上限 0.727
的语料上**一道题都没放宽过**，改前改后消融表一位数字都没变。所以这里重点测两件事：

1. 记录阶段真的把闸门要用的字段（分数 / 词面覆盖 / 判定来源）取回来了；
2. 扫描阶段只在"线上拿得到的行为"上做假设——精排是 mock、或这次压根没走精排分时，
   任何放宽线都不该改变判定，否则报告会承诺一个生产里不存在的配置。

模型加载与真实分数分布不在这里测，那是 `scripts/gate_probe.py` 现场跑出来的事。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gate_probe as gp
from rag_assistant.retrieval.models import RetrievedChunk, RetrievalResult


def chunk(doc_id: str = "a.md", score: float = 0.6) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{doc_id}-1",
        doc_id=doc_id,
        doc_name=doc_id,
        doc_type="policy",
        section_path="第一章",
        text="正文",
        rerank_score=score,
    )


def result(*, score: float, coverage: float, tokens: int, gate: str = "rerank", refused: bool) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        chunks=[chunk(score=score)],
        best_score=score,
        has_evidence=not refused,
        diagnostics={
            "gate": gate,
            "lexical_coverage": coverage,
            "content_tokens": [f"词{i}" for i in range(tokens)],
            "off_topic": (not refused and coverage < 0.3 and tokens > 0),
            "reason": "off_topic" if (not refused and coverage < 0.3 and tokens > 0) else "",
        },
    )


class FakeKb:
    def __init__(self, results: dict[str, RetrievalResult], status: dict[str, Any]) -> None:
        self._results = results
        self._status = status
        self.embedder = SimpleNamespace(name="BAAI/bge-m3", is_mock=status.get("embedder_mock", False))
        self.reranker = None if status.get("reranker") is None else SimpleNamespace(
            name=status["reranker"], is_mock=status.get("reranker_mock", False))

    def search(self, question: str) -> RetrievalResult:
        return self._results[question]

    def status(self) -> dict[str, Any]:
        return self._status


def install(monkeypatch: pytest.MonkeyPatch, kb: FakeKb) -> list[bool]:
    """把 ``gp.ChatService`` 换成返回这个 kb 的假服务，并记录 close 调用。"""
    closed: list[bool] = []

    class FakeService:
        def __init__(self, settings: Any) -> None:
            self.kb = kb

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(gp, "ChatService", FakeService)
    return closed


def golden(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path


BASE_STATUS = {
    "documents": 48,
    "vector_chunks": 782,
    "collection": "kb__bge-m3-1024",
    "embedder_mock": False,
    "reranker": "BAAI/bge-reranker-v2-m3",
    "reranker_mock": False,
    "raw_dir": "data/bench/raw",
}


def make_probe(**kwargs: Any) -> gp.Probe:
    payload = dict(
        question="q",
        variant="literal",
        domain="",
        should_refuse=False,
        best_score=0.60,
        gate="rerank",
        coverage=0.5,
        content_tokens=4,
        has_evidence=True,
        reason="",
        top_doc="a.md",
        n_chunks=1,
        reranker_mock=False,
    )
    payload.update(kwargs)
    return gp.Probe(**payload)


def counters(sim: dict[str, Any]) -> dict[str, int]:
    """扫描结果里带着 relax_line 本身（写表要用），断言时只看四个计数。"""
    return {key: sim[key] for key in ("false_reject", "answered", "correct_refuse", "leaked")}


# --------------------------------------------------------------------------
# 记录阶段
# --------------------------------------------------------------------------
def test_probe_all_records_what_the_gates_use(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = golden(tmp_path, [
        {"question": "可答题", "expected_doc": "a.md", "variant": "literal", "domain": "奖学金"},
        {"question": "天气题", "should_refuse": True},
    ])
    kb = FakeKb(
        {
            "可答题": result(score=0.6123, coverage=0.25, tokens=4, refused=False),
            "天气题": result(score=0.5210, coverage=0.0, tokens=3, refused=True),
        },
        BASE_STATUS,
    )
    closed = install(monkeypatch, kb)

    probes, env = gp.probe_all([path])

    assert closed == [True], "service 不关掉会把 Chroma 句柄漏出去"
    assert [(p.question, p.best_score, p.coverage, p.should_refuse) for p in probes] == [
        ("可答题", 0.6123, 0.25, False),
        ("天气题", 0.5210, 0.0, True),
    ]
    assert probes[0].content_tokens == 4
    assert probes[0].group == "可答/literal"
    assert probes[1].group == "应拒答"
    assert probes[0].top_doc == "a.md"
    # env 必须来自活着的 service，不能是另一份配置
    assert env["索引"] == "kb__bge-m3-1024｜782 chunk｜48 文档"
    assert env["Reranker"] == "BAAI/bge-reranker-v2-m3（mock=False）"
    assert env["score_threshold"] == pytest.approx(0.55)
    assert env["coverage_relax_score"] == pytest.approx(0.55)


def test_probe_all_flags_when_relaxation_is_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """mock 精排 / 关闭精排 / 没走精排分，三种情况都拿不到可信的 sigmoid 分。"""
    path = golden(tmp_path, [{"question": "q1", "expected_doc": "a.md"}])
    ok = {"q1": result(score=0.6, coverage=0.5, tokens=2, refused=False)}

    install(monkeypatch, FakeKb(ok, {**BASE_STATUS, "reranker_mock": True}))
    assert gp.probe_all([path])[0][0].relax_available is False

    # 关掉精排时检索层给的判定来源就不是 rerank，分数口径也换了（RRF/余弦），放宽不适用
    install(monkeypatch, FakeKb({"q1": result(score=0.6, coverage=0.5, tokens=2, gate="rrf", refused=False)},
                                {**BASE_STATUS, "reranker": None}))
    assert gp.probe_all([path])[0][0].relax_available is False
    assert gp.probe_all([path])[1]["Reranker"] == "关闭（mock=None）"

    install(monkeypatch, FakeKb({"q1": result(score=0.6, coverage=0.5, tokens=2, gate="vector_cosine", refused=False)},
                                BASE_STATUS))
    assert gp.probe_all([path])[0][0].relax_available is False


def test_probe_all_exits_when_index_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = golden(tmp_path, [{"question": "q1", "expected_doc": "a.md"}])
    closed = install(monkeypatch, FakeKb({"q1": result(score=0.6, coverage=0.5, tokens=2, refused=False)},
                                        {**BASE_STATUS, "vector_chunks": 0}))
    with pytest.raises(SystemExit) as excinfo:
        gp.probe_all([path])
    assert excinfo.value.code == 2
    assert closed == [True], "报错前也要把 service 关掉"


def test_probe_all_rejects_empty_golden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text("# 只有注释\n\n", encoding="utf-8")
    install(monkeypatch, FakeKb({}, BASE_STATUS))
    with pytest.raises(SystemExit):
        gp.probe_all([path])


# --------------------------------------------------------------------------
# 扫描阶段：判定顺序必须与 retriever 一致
# --------------------------------------------------------------------------
def test_dead_relax_line_reproduced() -> None:
    """0.75 那条线在真实分数（最高 0.727）上等于没写：所有放宽线给出同一个结果。"""
    probes = [make_probe(best_score=0.727, coverage=0.0), make_probe(best_score=0.51, coverage=0.0)]
    base = counters(gp.simulate(probes, threshold=0.55, min_coverage=0.30, relax_line=0.0))
    for line in (0.75, 0.80, 0.90):
        assert counters(gp.simulate(probes, threshold=0.55, min_coverage=0.30, relax_line=line)) == base


def test_relax_line_rescues_low_coverage_questions() -> None:
    probes = [
        make_probe(best_score=0.60, coverage=0.0),          # 可答、被词面闸门误拒
        make_probe(best_score=0.58, coverage=0.0),          # 刚好在 0.58 这条线上
        make_probe(best_score=0.52, coverage=0.0),          # 低于阈值：放宽线也救不了
        make_probe(best_score=0.60, coverage=0.0, should_refuse=True),  # 该拒的：放宽就漏
    ]
    strict = counters(gp.simulate(probes, threshold=0.55, min_coverage=0.30, relax_line=0.0))
    relaxed = counters(gp.simulate(probes, threshold=0.55, min_coverage=0.30, relax_line=0.58))

    # 三道可答题全部被拦（两道因为覆盖、一道连阈值都不够），该拒的那道拒对了
    assert strict == {"false_reject": 3, "answered": 0, "correct_refuse": 1, "leaked": 0}
    # 放宽线移到 0.58：两道误拒变答上，但那道该拒的也跟着漏了
    assert relaxed == {"false_reject": 1, "answered": 2, "correct_refuse": 0, "leaked": 1}


def test_relax_line_does_nothing_when_rerank_score_is_untrustworthy() -> None:
    """精排不可信时报告不许承诺放宽——扫描必须与 0 放宽线一致。"""
    probes = [make_probe(best_score=0.72, coverage=0.0, reranker_mock=True),
              make_probe(best_score=0.72, coverage=0.0, gate="vector_cosine")]
    assert counters(gp.simulate(probes, threshold=0.55, min_coverage=0.30, relax_line=0.55)) == \
        counters(gp.simulate(probes, threshold=0.55, min_coverage=0.30, relax_line=0.0))


def test_coverage_gate_is_skipped_entirely_when_floor_is_zero() -> None:
    probes = [make_probe(best_score=0.60, coverage=0.0), make_probe(best_score=0.60, coverage=0.0, should_refuse=True)]
    out = counters(gp.simulate(probes, threshold=0.55, min_coverage=0.0, relax_line=0.55))
    assert out == {"false_reject": 0, "answered": 1, "correct_refuse": 0, "leaked": 1}


def test_no_content_tokens_means_no_coverage_gate() -> None:
    """问句全是停用词时内容词为 0，闸门 2 不参与判定（与检索层一致）。"""
    out = gp.simulate([make_probe(best_score=0.60, coverage=0.0, content_tokens=0)],
                      threshold=0.55, min_coverage=0.30, relax_line=0.0)
    assert out["false_reject"] == 0


def test_empty_candidate_list_refuses_even_with_zero_threshold() -> None:
    """`has_evidence` 还有第三个条件：一路都没召回。重放漏掉它就会承诺线上不存在的答案。"""
    probe = make_probe(best_score=0.90, coverage=0.9, n_chunks=0)
    out = counters(gp.simulate([probe], threshold=0.0, min_coverage=0.0, relax_line=0.0))
    assert out == {"false_reject": 1, "answered": 0, "correct_refuse": 0, "leaked": 0}


def test_replay_check_catches_a_record_that_stopped_matching_the_retriever() -> None:
    """扫描表的可信度全靠这条：重放判定必须与实际 has_evidence 逐题一致。"""
    honest = [make_probe(best_score=0.60, coverage=0.5),
              make_probe(best_score=0.40, coverage=0.5, has_evidence=False)]
    assert gp.replay_mismatches(honest, threshold=0.55, min_coverage=0.30, relax_line=0.55) == []

    # 把其中一题的实际判定反过来（模拟"检索层加了新闸门但重放没跟上"）
    lying = [honest[0], make_probe(best_score=0.40, coverage=0.5, has_evidence=True, question="对不上的一题")]
    bad = gp.replay_mismatches(lying, threshold=0.55, min_coverage=0.30, relax_line=0.55)
    assert [p.question for p in bad] == ["对不上的一题"]


# --------------------------------------------------------------------------
# 分布描述
# --------------------------------------------------------------------------
def test_percentile_is_nearest_rank_without_interpolation() -> None:
    values = [0.10, 0.20, 0.30, 0.40]
    assert gp.percentile(values, 0.0) == 0.10
    assert gp.percentile(values, 0.25) == 0.20
    assert gp.percentile(values, 0.75) == 0.30
    assert gp.percentile(values, 1.0) == 0.40
    # 样本数为 3 时 p25 落在中间而不是 min：银行家舍入会在这里给出不同的答案
    assert gp.percentile([0.10, 0.20, 0.30], 0.25) == 0.20
    assert math.isnan(gp.percentile([], 0.5))


def test_describe_reports_counts_and_edges() -> None:
    text = gp.describe([0.5, 0.7, 0.6])
    assert text.startswith("n=3 min=0.500")
    assert "中位=0.600" in text
    assert text.endswith("max=0.700")
    assert gp.describe([]) == "-"


# --------------------------------------------------------------------------
# 渲染 / CLI
# --------------------------------------------------------------------------
@pytest.fixture
def meta() -> dict[str, Any]:
    return {
        "label": "unit",
        "题集": "golden.jsonl",
        "题数": 3,
        "可答": 2,
        "应拒答": 1,
        "Embedding": "BAAI/bge-m3（mock=False）",
        "Reranker": "BAAI/bge-reranker-v2-m3（mock=False）",
        "索引": "kb__bge-m3-1024｜782 chunk｜48 文档",
        "语料目录": "data/bench/raw",
        "score_threshold": 0.55,
        "min_lexical_coverage": 0.30,
        "coverage_relax_score": 0.55,
        "lines": [0.0, 0.55, 0.75],
    }


def test_render_marks_groups_and_the_current_default_line(meta: dict[str, Any]) -> None:
    probes = [
        make_probe(question="字面题", variant="literal", coverage=0.5, best_score=0.70),
        make_probe(question="口语题", variant="paraphrase", coverage=0.0, best_score=0.58, reason="off_topic"),
        make_probe(question="天气题", should_refuse=True, coverage=0.0, best_score=0.52),
    ]
    report = gp.render(probes, meta)

    assert "# 拒答闸门现场 · unit" in report
    assert "## 1. 分数分布" in report and "## 2. 放宽线扫描" in report
    assert "| 可答/literal |" in report and "| 可答/paraphrase |" in report and "| 应拒答 |" in report
    assert "## 3. 明细" in report
    assert "| 0.55 ← |" in report, "扫描表要把当前默认值标出来，否则读者不知道哪格是现配置"
    assert "| 0.75 |" in report
    assert "- 1 题，最优分区间 0.580–0.580" in report, "§4 要把被词面闸门误拒的题数与分数区间写出来"


def test_render_reports_the_replay_selfcheck(meta: dict[str, Any]) -> None:
    report = gp.render([make_probe(best_score=0.60, coverage=0.5)], meta)
    assert "逐题一致——上面这张扫描表可信" in report

    broken = gp.render([make_probe(best_score=0.40, coverage=0.5, has_evidence=True, question="对不上的一题")], meta)
    assert "有 **1 题**与检索层实际判定不一致" in broken
    assert "扫描数字先别拿去用" in broken


def test_render_reports_no_blocked_questions(meta: dict[str, Any]) -> None:
    report = gp.render([make_probe(coverage=0.9, best_score=0.70)], meta)
    assert "没有：词面闸门当前没有误拒可答题" in report


def test_cli_writes_report_and_rejects_missing_golden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = golden(tmp_path, [{"question": "q1", "expected_doc": "a.md"}])
    kb = FakeKb({"q1": result(score=0.60, coverage=0.10, tokens=3, refused=False)}, BASE_STATUS)
    install(monkeypatch, kb)
    out_path = tmp_path / "reports" / "probe.md"

    assert gp.main(["--golden", str(path), "--out", str(out_path), "--label", "cli"]) == 0
    text = out_path.read_text(encoding="utf-8")
    assert "# 拒答闸门现场 · cli" in text
    assert "kb__bge-m3-1024" in text

    assert gp.main(["--golden", str(tmp_path / "nope.jsonl")]) == 2
    assert "找不到题集" in capsys.readouterr().err


def test_cli_rejects_empty_lines_argument(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = golden(tmp_path, [{"question": "q1", "expected_doc": "a.md"}])
    install(monkeypatch, FakeKb({"q1": result(score=0.60, coverage=0.1, tokens=3, refused=False)}, BASE_STATUS))
    assert gp.main(["--golden", str(path), "--lines", " , "]) == 2
    assert "--lines 解析为空" in capsys.readouterr().err
