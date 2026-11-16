"""评测脚本测试：指标计算、阈值标定、报告渲染、端到端取数。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluate as ev
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.kb import KnowledgeBase


# --------------------------------------------------------------------------
# 评测集加载
# --------------------------------------------------------------------------
def test_load_golden_real_set(repo_root: Path) -> None:
    items = ev.load_golden(repo_root / "data" / "eval" / "golden_set.jsonl")
    assert len(items) >= 30
    assert sum(1 for item in items if item.should_refuse) >= 2
    assert all(item.question for item in items)
    answered = [item for item in items if not item.should_refuse]
    assert all(item.expected_doc for item in answered)
    assert all(item.expected_answer_points for item in answered)


def test_load_golden_skips_blank_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        "# 注释\n\n"
        + json.dumps({"question": "Q1", "expected_doc": "a.md"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    items = ev.load_golden(path)
    assert len(items) == 1
    assert items[0].expected_doc == "a.md"


def test_load_golden_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ev.load_golden(path)


def test_load_golden_requires_question(tmp_path: Path) -> None:
    path = tmp_path / "noq.jsonl"
    path.write_text('{"expected_doc": "a.md"}\n', encoding="utf-8")
    with pytest.raises(SystemExit):
        ev.load_golden(path)


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def make_result(
    *,
    should_refuse: bool = False,
    rank: int | None = 1,
    refuted: bool = False,
    best_score: float = 0.6,
    points: dict[str, bool] | None = None,
    faithful: bool | None = True,
    latency: float = 100.0,
) -> ev.ItemResult:
    return ev.ItemResult(
        item=ev.GoldenItem(question="q", expected_doc="a.md", should_refuse=should_refuse),
        answer="answer",
        refused=refuted,
        refusal_reason=None,
        best_score=best_score,
        rank=rank,
        citations=[],
        cited_texts=[],
        point_hits=points or {},
        faithful=faithful,
        judge_reason="",
        judge="规则",
        latency_ms=latency,
        usage_prompt=0,
        usage_completion=0,
    )


def test_metrics_hit_and_mrr() -> None:
    results = [
        make_result(rank=1),
        make_result(rank=3),
        make_result(rank=None),  # 未命中
        make_result(should_refuse=True, rank=None, refuted=True),
        make_result(should_refuse=True, rank=None, refuted=True),
    ]
    metrics = ev.compute_metrics(results)

    assert metrics["n"] == 5
    assert metrics["n_answerable"] == 3
    assert metrics["hit@1"] == pytest.approx(1 / 3)
    assert metrics["hit@3"] == pytest.approx(2 / 3)
    assert metrics["hit@5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert metrics["refusal_recall"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0


def test_metrics_false_reject_and_missed_refusal() -> None:
    """误拒（可答但拒）与漏答（该拒却答）都要反映在指标里。"""
    results = [
        make_result(refuted=True),  # 误拒
        make_result(rank=1),
        make_result(should_refuse=True, refuted=False),  # 漏答
        make_result(should_refuse=True, refuted=True),
    ]
    metrics = ev.compute_metrics(results)

    assert metrics["false_reject_rate"] == pytest.approx(0.5)
    assert metrics["refusal_recall"] == pytest.approx(0.5)
    assert metrics["refusal_precision"] == pytest.approx(0.5)
    assert metrics["refusal_accuracy"] == pytest.approx(0.5)


def test_metrics_point_coverage_and_faithfulness() -> None:
    results = [
        make_result(points={"一年为期": True, "累计两年": False}, faithful=False),
        make_result(points={"15 个工作日": True}, faithful=True),
    ]
    metrics = ev.compute_metrics(results)
    assert metrics["point_coverage"] == pytest.approx(2 / 3)
    assert metrics["faithfulness"] == pytest.approx(0.5)


def test_metrics_without_judgement() -> None:
    metrics = ev.compute_metrics([make_result(faithful=None)])
    assert metrics["faithfulness"] is None


def test_metrics_latency_percentiles() -> None:
    results = [make_result(latency=float(value)) for value in (10, 20, 30, 40, 50, 200)]
    metrics = ev.compute_metrics(results)
    assert metrics["latency_p50_ms"] == 30.0
    # 只有 6 个样本时，最近秩法（nearest-rank）的 P95 就是最大值
    assert metrics["latency_p95_ms"] == 200.0


# --------------------------------------------------------------------------
# 阈值标定
# --------------------------------------------------------------------------
def test_sweep_and_recommendation() -> None:
    results = [
        make_result(best_score=0.72),
        make_result(best_score=0.45),
        make_result(should_refuse=True, rank=None, refuted=True, best_score=0.11),
        make_result(should_refuse=True, rank=None, refuted=True, best_score=0.05),
    ]
    rows = ev.sweep_thresholds(results)
    assert rows[0]["threshold"] == 0.05

    row_50 = next(row for row in rows if row["threshold"] == 0.5)
    assert row_50["false_reject"] == 1  # 0.45 被误拒
    assert row_50["correct_refuse"] == 2

    recommended = ev.recommend_threshold(rows)
    assert recommended is not None
    # 0.11 < t <= 0.45 时：两题可答都通过、两题应拒都拒
    assert 0.11 < recommended["threshold"] <= 0.45
    assert recommended["accuracy"] == 1.0


def test_sweep_empty() -> None:
    assert ev.sweep_thresholds([])
    assert ev.recommend_threshold([]) is None


# --------------------------------------------------------------------------
# 判定工具
# --------------------------------------------------------------------------
def test_point_coverage_ignores_spaces() -> None:
    hits = ev._point_coverage(["累计不得超过两年", "工本费 2 元"], "休学期一般以一年为期，累计不得超过 两年。工本费2元")
    assert hits["累计不得超过两年"] is True
    assert hits["工本费 2 元"] is True


def test_judge_by_rule() -> None:
    assert ev._judge_by_rule({"a": True}, True)[0] is True
    ok, reason = ev._judge_by_rule({"a": True, "b": False}, True)
    assert ok is False and "b" in reason
    assert ev._judge_by_rule({}, False)[0] is False


def test_judge_by_llm_skips_mock() -> None:
    from rag_assistant.llm import MockLLM

    faithful, reason = ev._judge_by_llm(MockLLM(), ev.GoldenItem(question="q"), "a", ["t"])
    assert faithful is None
    assert "mock" in reason


# --------------------------------------------------------------------------
# 报告渲染
# --------------------------------------------------------------------------
def test_render_report_contains_sections() -> None:
    results = [make_result(points={"一年为期": True})]
    metrics = ev.compute_metrics(results)
    rows = ev.sweep_thresholds(results)
    report = ev.render_report(
        label="unit-test",
        metrics=metrics,
        results=results,
        sweep=rows,
        recommended=ev.recommend_threshold(rows),
        environment={"LLM": "mock-llm（mock=True）"},
        threshold_used=0.5,
        judge_mode="规则要点覆盖",
    )
    for heading in ("# RAG 评测报告 · unit-test", "## 1. 环境", "## 2. 总体指标", "## 3. 逐题结果", "## 4. 阈值标定", "## 6. 口径与限制"):
        assert heading in report
    assert "LLM: mock-llm" not in report  # 表格式渲染，不是 YAML
    assert "推荐阈值" in report


# --------------------------------------------------------------------------
# 端到端取数（mock 链路 + 真实 Chroma）
# --------------------------------------------------------------------------
def test_evaluate_item_end_to_end(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    service = ChatService(mock_settings, kb=kb)
    try:
        answerable = ev.GoldenItem(
            question="在读证明在哪里办、多久能拿到？",
            expected_doc="办事指南-在读证明.txt",
            expected_answer_points=["1 个工作日"],
        )
        # 阈值传 0：本用例验证的是"评测取数与判定链路"，不是 mock 分数分布
        result = ev.evaluate_item(service, answerable, judge_mode="rule", threshold=0.0)
        assert result.rank is not None, "期望文档应出现在候选里"
        assert result.citations, "应带引用"
        assert result.faithful is True

        refusing = ev.GoldenItem(question="食堂在哪个楼，几点开饭？", should_refuse=True)
        refused = ev.evaluate_item(service, refusing, judge_mode="rule", threshold=0.0)
        assert refused.refused is True
        assert refused.faithful is True
    finally:
        service.close()


# --------------------------------------------------------------------------
# 引用 → 资料块映射（曾经的真 bug）
# --------------------------------------------------------------------------
def test_cited_contexts_maps_by_section_not_by_renumbered_index() -> None:
    """引用重编号后 index 不再等于资料块编号，必须按 (doc_id, section_path) 取正文。"""
    from rag_assistant.generation.answerer import AnswerResult
    from rag_assistant.generation.citation import Citation

    record = AnswerResult(question="q", answer="a")
    record.context_blocks = [
        {"index": 1, "doc_id": "d1", "chunk_id": "d1#0", "doc_name": "a.md", "section_path": "一", "body": "正文一"},
        {"index": 2, "doc_id": "d2", "chunk_id": "d2#0", "doc_name": "b.md", "section_path": "四、所需材料", "body": "正文二：委托书"},
    ]
    # 答案只引了资料块 2 → 重编号后 citation.index=1
    record.citations = [
        Citation(
            index=1,
            doc_id="d2",
            doc_name="b.md",
            section_path="四、所需材料",
            page=0,
            chunk_id="d2#0",
            quote="正文二：委托书",
        )
    ]
    citations, texts = ev._cited_contexts(record)
    assert citations[0]["source_text"] == "正文二：委托书"
    assert texts == ["正文二：委托书"]
