"""运维台：实时计数、质量口径、错误日志与接口形状。

这些用例不依赖真实模型：`mock_settings` 已经把 llm / embedding / reranker 换成 mock，
向量库仍是真实 Chroma，所以接口拿到的是"能跑起来的最小真相"。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import create_app
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.insights.live import LiveTracker
from rag_assistant.insights.ops import (
    human_size,
    latest_eval_summary,
    quality_report,
    recent_events,
)
from rag_assistant.kb import KnowledgeBase
from rag_assistant.logging_setup import setup_logging


# --------------------------------------------------------------------------
# 实时计数器
# --------------------------------------------------------------------------
def test_live_tracker_counts_online_and_inflight() -> None:
    tracker = LiveTracker(window_seconds=300)
    tracker.begin("alice::s1")
    tracker.begin("alice::s1")
    snapshot = tracker.snapshot()
    assert snapshot["online"] == 1
    assert snapshot["inflight"] == 2

    tracker.end()
    tracker.end()
    snapshot = tracker.snapshot()
    assert snapshot["inflight"] == 0
    # 两个会话各问过一次，但只有 alice 一个人在线
    assert snapshot["online"] == 1
    assert snapshot["peak_online"] == 1
    assert snapshot["questions_per_minute"] == 2


def test_live_tracker_drops_stale_sessions() -> None:
    tracker = LiveTracker(window_seconds=0.05)
    tracker.touch("bob::s2")
    assert tracker.snapshot()["online"] == 1
    time.sleep(0.12)
    assert tracker.snapshot()["online"] == 0


def test_live_tracker_records_latency_percentiles() -> None:
    tracker = LiveTracker()
    tracker.begin("carol::s3")
    tracker.end(latency_ms=120.0)
    tracker.begin("carol::s3")
    tracker.end(latency_ms=980.0)
    latency = tracker.snapshot()["latency_ms"]
    assert latency["samples"] == 2
    assert latency["p50"] in {120.0, 980.0}
    assert latency["p95"] == 980.0


def test_live_tracker_keeps_recent_errors() -> None:
    tracker = LiveTracker()
    tracker.record_error("LLMError", "upstream 502")
    errors = tracker.snapshot()["errors_recent"]
    assert errors[-1]["kind"] == "LLMError"
    assert "502" in errors[-1]["detail"]


# --------------------------------------------------------------------------
# 质量口径
# --------------------------------------------------------------------------
def _record(question: str, *, refused: bool, reason: str | None = None, **extra) -> dict:
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "refused": refused,
        "refusal_reason": reason,
        "latency_ms": 800.0,
    }
    payload.update(extra)
    return payload


def test_quality_report_separates_answer_from_citation() -> None:
    records = [
        _record("休学最长多久", refused=False, retrieved_ids=["c1"], best_score=0.71),
        _record("复学要带什么", refused=False, retrieved_ids=[], best_score=0.66),
        _record("郑州天气", refused=True, reason="off_topic"),
        _record("食堂几点开饭", refused=True, reason="below_threshold"),
    ]
    report = quality_report(records, days=7)
    assert report["total"] == 4
    assert report["answered"] == 2
    assert report["refused"] == 2
    # 应答率只说明"没有拒答"，引用覆盖率只统计作答里的
    assert report["answer_rate"] == 0.5
    assert report["with_citation"] == 1
    assert report["citation_rate"] == 0.5
    labels = {row["reason"]: row["label"] for row in report["reasons"]}
    assert labels["off_topic"] == "问题不在知识库范围"
    assert {row["question"] for row in report["top_refused"]} == {"郑州天气", "食堂几点开饭"}


def test_quality_report_fills_empty_days() -> None:
    report = quality_report([], days=3)
    assert len(report["daily"]) == 3
    assert all(row["total"] == 0 for row in report["daily"])
    assert report["answer_rate"] == 0.0
    assert report["latency_ms"]["samples"] == 0


def test_quality_report_uses_claims_support_when_present() -> None:
    records = [
        _record("休学最长多久", refused=False, retrieved_ids=["c1"], claims_support=0.8),
        _record("复学要带什么", refused=False, retrieved_ids=["c2"], claims_support=1.0),
    ]
    report = quality_report(records, days=1)
    assert report["claims_samples"] == 2
    assert report["avg_claims_support"] == pytest.approx(0.9)


def test_quality_report_recent_latency_ignores_old_slow_samples() -> None:
    """整窗 P95 会被早期几百秒的记录拉高，最近 N 次要能反映当前手感。"""
    records = [_record(f"old{i}", refused=False, retrieved_ids=["c"], latency_ms=300000.0) for i in range(10)]
    records += [_record(f"new{i}", refused=False, retrieved_ids=["c"], latency_ms=1500.0) for i in range(50)]
    report = quality_report(records, days=1)
    assert report["latency_ms"]["samples"] == 60
    assert report["latency_ms"]["p95"] > 60000
    recent = report["latency_recent"]
    assert recent["samples"] == 50
    assert recent["p50"] == 1500.0
    assert recent["p95"] == 1500.0


# --------------------------------------------------------------------------
# 日志与体积
# --------------------------------------------------------------------------
def test_recent_events_only_keeps_errors_and_warnings(tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text(
        "\n".join(
            [
                "2026-11-15 10:00:00 | INFO | rag_assistant.api | 服务启动",
                "2026-11-15 10:00:01 | WARNING | rag_assistant.api | 示例口令",
                "2026-11-15 10:00:02 | ERROR | rag_assistant.api | LLM 调用失败",
                "随便一行没有格式的内容",
            ]
        ),
        encoding="utf-8",
    )
    events = recent_events(log, limit=10)
    assert [item["level"] for item in reversed(events)] == ["WARNING", "ERROR"]
    assert "LLM 调用失败" in events[0]["message"]


def test_latest_eval_summary_reads_newest_report(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    older = reports / "eval-old.md"
    older.write_text("判断准确率 80.0%\n", encoding="utf-8")
    newer = reports / "eval-new.md"
    newer.write_text(
        "| 判断准确率 | 90.5% |\n| 被诱导成功率 | 0% |\n| Hit@1 | 31/31 |\n",
        encoding="utf-8",
    )
    now = time.time()
    os.utime(older, (now - 3600, now - 3600))
    os.utime(newer, (now, now))

    summary = latest_eval_summary(reports)
    assert summary is not None
    assert summary["file"] == "eval-new.md"
    assert summary["metrics"]["decision_accuracy"] == "90.5%"
    assert summary["metrics"]["hit_at_1"] == "31/31"


def test_latest_eval_summary_without_reports(tmp_path: Path) -> None:
    assert latest_eval_summary(tmp_path / "missing") is None


def test_human_size() -> None:
    assert human_size(0) == "0 B"
    assert human_size(1536).endswith("KB")
    assert human_size(5 * 1024 * 1024) == "5.0 MB"


# --------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------
@pytest.fixture()
def console_client(mock_settings: Settings, sample_raw_dir: Path):
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    service = ChatService(mock_settings, kb=kb)
    # setup_logging 有全局幂等标志，不强制重配的话 QA 日志会落在前一个用例的目录里
    setup_logging(mock_settings, force=True)
    app = create_app(mock_settings, chat_service=service)
    with TestClient(app) as test_client:
        yield test_client


def test_console_live_has_zero_before_any_question(console_client: TestClient) -> None:
    response = console_client.get("/console/live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["online"] == 0
    assert payload["inflight"] == 0
    assert payload["single_process"] is True
    assert len(payload["series"]) == 60
    assert payload["brand"]["school_name"]


def test_console_live_counts_a_question(console_client: TestClient) -> None:
    console_client.post("/chat", json={"session_id": "ops-1", "question": "休学最长可以多久？"})
    payload = console_client.get("/console/live").json()
    assert payload["online"] >= 1
    assert payload["questions_per_minute"] >= 1
    assert payload["latency_ms"]["samples"] >= 1


def test_console_quality_reads_the_qa_log(console_client: TestClient, mock_settings: Settings) -> None:
    console_client.post("/chat", json={"session_id": "ops-2", "question": "休学最长可以多久？"})
    console_client.post("/chat", json={"session_id": "ops-3", "question": "郑州今天天气怎么样？"})
    payload = console_client.get("/console/quality?days=7").json()
    assert payload["total"] >= 2
    assert payload["refused"] >= 1
    assert payload["window_days"] == 7
    assert "eval" in payload
    assert payload["notes"]["eval"]


def test_console_system_reports_models_and_index(console_client: TestClient) -> None:
    payload = console_client.get("/console/system").json()
    assert payload["index"]["documents"] >= 4
    assert payload["index"]["vector_chunks"] > 0
    assert payload["models"]["llm_mock"] is True
    assert payload["models"]["embedder"]
    assert payload["background"]["sync_interval_seconds"] >= 0
    assert payload["paths"]["index_dir"]
    assert payload["uptime_seconds"] >= 0


def test_console_events_tolerates_missing_log(console_client: TestClient) -> None:
    payload = console_client.get("/console/events").json()
    assert "events" in payload
    assert isinstance(payload["events"], list)


def test_console_top_returns_question_rows(console_client: TestClient) -> None:
    console_client.post("/chat", json={"session_id": "ops-4", "question": "图书馆几点开门？"})
    payload = console_client.get("/console/top?days=7&limit=5").json()
    assert payload["window_days"] == 7
    assert isinstance(payload["top_questions"], list)
    assert isinstance(payload["departments"], list)


def test_console_sync_triggers_a_job(console_client: TestClient) -> None:
    response = console_client.post("/console/sync")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"running", "queued", "busy", "success"}
    assert "job_id" in payload


def test_console_page_is_served(console_client: TestClient) -> None:
    response = console_client.get("/console")
    assert response.status_code == 200
    assert "运维台" in response.text
    assert "/console/live" in response.text
