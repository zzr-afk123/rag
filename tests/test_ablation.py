"""消融脚本的测试：不跑真链路，只验证"每一行确实改了配置、表确实会算错"。

消融表是最容易自欺的东西：`--config` 拼错一个字母、或者覆盖根本没生效，
出来的五行数字会长得跟"完美消融"一模一样，而没人会怀疑。所以这里把
"行 → 实际生效的字段值"当作断言对象，而不是去断言模型分数。
"""

from __future__ import annotations

from typing import Any

import pytest

import ablation as ab
import evaluate as ev
from rag_assistant.config import Settings, get_settings


def fake_item(question: str, *, refuse: bool = False, variant: str = "literal", domain: str = "") -> ev.GoldenItem:
    return ev.GoldenItem(
        question=question,
        expected_doc="" if refuse else "a.md",
        should_refuse=refuse,
        variant=variant,
        domain=domain,
    )


def fake_result(item: ev.GoldenItem, *, rank: int | None, answer_rank: int | None, refused: bool) -> ev.ItemResult:
    return ev.ItemResult(
        item=item,
        answer="答案",
        refused=refused,
        refusal_reason=None,
        best_score=0.4,
        rank=rank,
        answer_rank=answer_rank,
        citations=[],
        cited_texts=[],
        point_hits={"要点": not refused},
        faithful=None,
        judge_reason="",
        judge="规则",
        latency_ms=100.0,
        usage_prompt=0,
        usage_completion=0,
    )


# --------------------------------------------------------------------------
# 行的配置覆盖
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("row_name", "expectations"),
    [
        ("BM25 only", {"retrieval.weight_vector": 0.0, "retrieval.weight_bm25": 1.0}),
        ("向量 only", {"retrieval.weight_vector": 1.0, "retrieval.weight_bm25": 0.0}),
        ("+RRF", {"reranker.enabled": False}),
        ("+rerank", {}),
        ("关词面闸门", {"retrieval.min_lexical_coverage": 0.0}),
        ("关闸门", {"retrieval.min_lexical_coverage": 0.0, "generation.claim_check": "off"}),
    ],
)
def test_each_row_changes_only_what_it_claims(row_name: str, expectations: dict[str, Any]) -> None:
    base = get_settings()
    row = next(item for item in ab.ROWS if item.name == row_name)
    settings, applied = ev.apply_overrides(base, row.overrides)

    for dotted, expected in expectations.items():
        node: Any = settings
        for part in dotted.split("."):
            node = getattr(node, part)
        assert node == expected, f"{row_name} 的 {dotted} 没生效"

    # 除"+rerank"是默认配置外，其余行必须真的改动过东西
    if row_name != "+rerank":
        assert applied, f"{row_name} 没产生任何覆盖，这行数字是假的"
    # 没在参数里声明的旋钮必须保持默认，否则"这行只改了 X"这句话就是假的
    if "generation.claim_check" not in expectations:
        assert settings.generation.claim_check == base.generation.claim_check
    if "retrieval.min_lexical_coverage" not in expectations:
        assert settings.retrieval.min_lexical_coverage == base.retrieval.min_lexical_coverage


def test_rows_are_named_as_the_readme_table_claims() -> None:
    assert [row.name for row in ab.ROWS if row.kind == "main"] == [
        "BM25 only",
        "向量 only",
        "+RRF",
        "+rerank",
        "关词面闸门",
        "关闸门",
    ]
    # 实验档不进主表，README 的数字不许引用它们
    assert all(row.kind == "probe" for row in ab.ROWS if row.kind != "main")


def test_refusal_gate_row_runs_with_threshold_zero() -> None:
    """关闸门行必须显式把阈值压到 0：否则 evaluate_item 仍按默认阈值拒答，闸门没关。"""
    row = next(item for item in ab.ROWS if item.name == "关闸门")
    assert row.threshold == 0.0
    assert all(item.threshold is None for item in ab.ROWS if item.name == "+rerank")
    assert all(row.threshold is None for row in ab.ROWS if row.kind == "main" and row.name != "关闸门")


# --------------------------------------------------------------------------
# run_row（用假 service，只验装配）
# --------------------------------------------------------------------------
def test_run_row_reports_effective_config(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class FakeService:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings
            self.llm = None

        def close(self) -> None:
            closed.append(True)

    def fake_evaluate_item(service: Any, item: ev.GoldenItem, **_: Any) -> ev.ItemResult:
        return fake_result(item, rank=1, answer_rank=None if item.should_refuse else 2, refused=item.should_refuse)

    monkeypatch.setattr(ab, "ChatService", FakeService)
    monkeypatch.setattr(ev, "evaluate_item", fake_evaluate_item)

    row = next(item for item in ab.ROWS if item.name == "向量 only")
    out = ab.run_row(
        get_settings(),
        row,
        [fake_item("q1"), fake_item("q2", refuse=True)],
        judge_mode="none",
    )

    assert closed == [True], "service 必须关掉，否则 Chroma 句柄会漏"
    assert out["applied"] == ["retrieval.weight_bm25=0"]
    assert out["effective"]["weight_bm25"] == 0.0
    assert out["effective"]["reranker"] is True
    assert out["metrics"]["n"] == 2
    assert out["metrics"]["answer_hit@1"] == 0.0  # 答案里引的是第 2 个


def test_run_row_threshold_is_passed_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    class FakeService:
        def __init__(self, settings: Settings) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_evaluate_item(service: Any, item: ev.GoldenItem, **kwargs: Any) -> ev.ItemResult:
        seen.append(kwargs["threshold"])
        return fake_result(item, rank=1, answer_rank=1, refused=False)

    monkeypatch.setattr(ab, "ChatService", FakeService)
    monkeypatch.setattr(ev, "evaluate_item", fake_evaluate_item)
    base = get_settings()
    for name in ("+rerank", "关闸门"):
        row = next(item for item in ab.ROWS if item.name == name)
        ab.run_row(base, row, [fake_item("q")], judge_mode="none")
    assert seen == [None, 0.0]


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def build_rows() -> list[dict[str, Any]]:
    metrics = ev.compute_metrics([fake_result(fake_item("q"), rank=1, answer_rank=1, refused=False)])
    metrics.update({"n": 2, "n_refuse": 1, "refusal_recall": 1.0})
    rows = []
    for name in ("+rerank", "关闸门"):
        row = next(item for item in ab.ROWS if item.name == name)
        rows.append(
            {
                "name": name,
                "purpose": row.purpose,
                "applied": list(row.overrides),
                "effective": {
                    "weight_vector": 1.0,
                    "weight_bm25": 0.0,
                    "reranker": True,
                    "min_lexical_coverage": 0.0,
                    "coverage_relax_score": 0.55,
                    "score_threshold": row.threshold,
                    "claim_check": "off",
                },
                "metrics": metrics,
            }
        )
    return rows


def test_render_marks_missing_denominators_as_dash() -> None:
    """没有应拒题的行不许显示"正确拒答率 0.0%"，那读起来像闸门全废。"""
    base = get_settings()
    row = next(item for item in ab.ROWS if item.name == "+rerank")
    metrics = ev.compute_metrics([fake_result(fake_item("q"), rank=1, answer_rank=1, refused=False)])
    cell = ab._cell(row.name, metrics)
    assert cell.endswith("| - | 100ms |")
    assert "| 100.0% |" in cell


def test_render_output_shape() -> None:
    report = ab.render(build_rows(), [fake_item("q", variant="literal")], {"label": "unit", "题数": 1})
    assert "# 消融表 · unit" in report
    assert "## 1. 总体" in report
    assert "## 2. 每行实际生效的配置" in report
    assert "## 3. 按改写强度拆开" in report
    assert "按领域拆开" not in report  # domain 全空时不印空表


def test_render_no_group_tables_when_unlabeled() -> None:
    items = [ev.GoldenItem(question="q", expected_doc="a.md")]
    report = ab.render(build_rows(), items, {"label": "plain"})
    assert "## 3." not in report


def test_cli_rejects_unknown_only_row(capsys: pytest.CaptureFixture) -> None:
    assert ab.main(["--only", "不存在的行"]) == 2
    assert "没匹配到任何行" in capsys.readouterr().err
