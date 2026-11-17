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


def test_load_golden_reads_variant_and_domain(tmp_path: Path) -> None:
    """两套题集靠 variant 区分，字段读不到就等于消融表少一列。"""
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps({"question": "缓交学费几天？", "expected_doc": "a.md",
                    "variant": "literal", "domain": "财务"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"question": "我先把学上着，钱后面补行不行？", "expected_doc": "a.md",
                      "variant": "paraphrase", "domain": "财务"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"question": "老规矩", "expected_doc": "a.md"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    literal, paraphrase, unlabeled = ev.load_golden(path)
    assert (literal.variant, literal.domain) == ("literal", "财务")
    assert paraphrase.variant == "paraphrase"
    assert (unlabeled.variant, unlabeled.domain) == ("", "")


# --------------------------------------------------------------------------
# 配置覆盖（消融表每一行都得可复现）
# --------------------------------------------------------------------------
def test_apply_overrides_coerces_by_current_type(mock_settings: Settings) -> None:
    settings, applied = ev.apply_overrides(
        mock_settings,
        [
            "retrieval.weight_bm25=0",
            "reranker.enabled=false",
            "retrieval.score_threshold=0.85",
            "generation.claim_check=off",
        ],
    )
    assert settings.retrieval.weight_bm25 == 0.0
    assert settings.reranker.enabled is False
    assert settings.retrieval.score_threshold == pytest.approx(0.85)
    assert settings.generation.claim_check == "off"
    assert applied == [
        "retrieval.weight_bm25=0",
        "reranker.enabled=false",
        "retrieval.score_threshold=0.85",
        "generation.claim_check=off",
    ]


def test_apply_overrides_does_not_mutate_the_original(mock_settings: Settings) -> None:
    """frozen dataclass + replace：原对象必须保持默认值，否则同进程里前后两轮会互相污染。"""
    before = mock_settings.retrieval.weight_vector
    settings, _ = ev.apply_overrides(mock_settings, ["retrieval.weight_vector=0", "sync.include_patterns=*.md"])
    assert settings.retrieval.weight_vector == 0.0
    assert settings.sync.include_patterns == ("*.md",)
    assert mock_settings.retrieval.weight_vector == before


def test_apply_overrides_rejects_bad_input(mock_settings: Settings) -> None:
    for text in (
        "retrieval.weight_bm25",          # 没有 =
        "weight_bm25=0",                  # 少一层 section
        "retrieval.nope=1",               # 字段不存在
        "reranker.enabled=maybe",         # 布尔写错
        "retrieval.top_n=abc",            # 整数写错
        "retrieval.score_threshold=x",    # 小数写错
    ):
        with pytest.raises(SystemExit):
            ev.apply_overrides(mock_settings, [text])


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def make_result(
    *,
    should_refuse: bool = False,
    rank: int | None = 1,
    answer_rank: int | None = None,
    refuted: bool = False,
    best_score: float = 0.6,
    points: dict[str, bool] | None = None,
    faithful: bool | None = True,
    latency: float = 100.0,
    variant: str = "",
    domain: str = "",
) -> ev.ItemResult:
    return ev.ItemResult(
        item=ev.GoldenItem(
            question="q", expected_doc="a.md", should_refuse=should_refuse, variant=variant, domain=domain
        ),
        answer="answer",
        refused=refuted,
        refusal_reason=None,
        best_score=best_score,
        rank=rank,
        answer_rank=answer_rank,
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
# 答案正文口径的 Hit（"检索捞到了但没引"必须算失败）
# --------------------------------------------------------------------------
def test_answer_hit_penalises_uncited_evidence() -> None:
    """rank 命中、答案却没引用期望文档 → 候选 Hit@1 满分，答案 Hit@1 归零。"""
    results = [
        make_result(rank=1, answer_rank=None),
        make_result(rank=1, answer_rank=2),
        make_result(rank=2, answer_rank=1),
    ]
    metrics = ev.compute_metrics(results)

    assert metrics["hit@1"] == pytest.approx(2 / 3)
    assert metrics["answer_hit@1"] == pytest.approx(1 / 3)
    assert metrics["answer_hit@3"] == pytest.approx(2 / 3)
    # 未命中的那题不贡献倒数，答案 MRR = (1/2 + 1)/2
    assert metrics["answer_mrr"] == pytest.approx(0.75)
    # 两套口径不能互相污染
    assert metrics["mrr"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_answer_rank_reads_citation_index() -> None:
    from rag_assistant.generation.answerer import AnswerResult
    from rag_assistant.generation.citation import Citation

    def citation(index: int, doc_name: str) -> Citation:
        return Citation(
            index=index, doc_id=doc_name, doc_name=doc_name, section_path="第一条", page=0,
            chunk_id=f"{doc_name}#0", quote="略",
        )

    record = AnswerResult(question="q", answer="a")
    record.citations = [citation(1, "a.md"), citation(2, "b.md")]
    assert ev._answer_rank_of_expected(record, {"b.md"}) == 2
    assert ev._answer_rank_of_expected(record, {"c.md"}) is None
    # 没有期望文档（应拒答题）时不参与该口径
    assert ev._answer_rank_of_expected(record, set()) is None


def test_answer_rank_falls_back_to_position_when_index_is_missing() -> None:
    """异常数据里 index 为 0 也不能丢掉名次信息。"""
    from rag_assistant.generation.answerer import AnswerResult
    from rag_assistant.generation.citation import Citation

    record = AnswerResult(question="q", answer="a")
    record.citations = [
        Citation(index=0, doc_id="a.md", doc_name="a.md", section_path="", page=0, chunk_id="a#0", quote="")
    ]
    assert ev._answer_rank_of_expected(record, {"a.md"}) == 1


# --------------------------------------------------------------------------
# 分组（字面改写 / 他人重写）
# --------------------------------------------------------------------------
def test_metrics_split_by_variant() -> None:
    results = [
        make_result(rank=1, answer_rank=1, variant="literal"),
        make_result(rank=1, answer_rank=None, variant="literal"),
        make_result(rank=5, answer_rank=None, variant="paraphrase"),
        make_result(rank=2, answer_rank=1, variant="paraphrase"),
    ]
    groups = {row["name"]: row for row in ev.compute_metrics(results)["by_variant"]}

    assert set(groups) == {"literal", "paraphrase"}
    assert groups["literal"]["n_answerable"] == 2
    assert groups["literal"]["hit@1"] == 1.0
    assert groups["literal"]["answer_hit@1"] == 0.5
    assert groups["paraphrase"]["hit@1"] == 0.0
    assert groups["paraphrase"]["answer_hit@1"] == 0.5
    # 未标注不能混进这两组：单独一份 golden 里 variant 缺省时归 unlabeled
    unlabeled = ev.compute_metrics([make_result()])["by_variant"]
    assert [row["name"] for row in unlabeled] == ["unlabeled"]


def test_metrics_split_by_domain() -> None:
    rows = ev.compute_metrics([make_result(domain="学籍"), make_result(domain="财务")])["by_domain"]
    # sorted() 按码位排，中文不是拼音序：断言集合而不是顺序，报告里再管排序
    assert {row["name"] for row in rows} == {"学籍", "财务"}


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


def test_render_report_shows_answer_hit_and_groups() -> None:
    """两套口径 + 分组表要真出现在报告里，不然只是算完了没人看。"""
    results = [
        make_result(rank=1, answer_rank=None, variant="literal", domain="财务"),
        make_result(rank=1, answer_rank=1, variant="paraphrase", domain="财务"),
    ]
    metrics = ev.compute_metrics(results)
    report = ev.render_report(
        label="group",
        metrics=metrics,
        results=results,
        sweep=ev.sweep_thresholds(results),
        recommended=None,
        environment={"LLM": "mock"},
        threshold_used=0.5,
        judge_mode="规则要点覆盖",
    )
    assert "| **答案 Hit@1** |" in report
    assert "## 3.2 分组指标" in report
    assert "| paraphrase | 1 | 1 |" in report
    assert "| 财务 | 2 | 2 |" in report
    assert "候选名次" in report and "答案引" in report


def test_render_report_omits_unlabeled_group_tables() -> None:
    """题集没标 variant/domain 时不该印两张全 unlabeled 的空表刷存在感。"""
    results = [make_result()]
    report = ev.render_report(
        label="plain",
        metrics=ev.compute_metrics(results),
        results=results,
        sweep=ev.sweep_thresholds(results),
        recommended=None,
        environment={"LLM": "mock"},
        threshold_used=0.5,
        judge_mode="规则要点覆盖",
    )
    assert "## 3.2 分组指标" in report
    assert "### 按改写强度" not in report


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
        # 答案正文口径：mock 链路只引那一块，序号必须是 1
        assert result.answer_rank == 1
        assert result.answer_hit_at[1] is True

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
