"""知识库缺口清单（补料清单）与"引用回查原文"的测试。

覆盖点：
- 只统计被拒答的问题，按"部门 + 归一化问句"聚合，按被问次数排序；
- 同一条问句的不同说法（标点/空格差异）算同一条，但保留最多 3 种问法；
- 没能识别部门的问题单独成段，不丢；
- Markdown 结构（标题、部门名、次数、"建议"）与演示数据警告；
- ``KnowledgeBase.get_chunks`` 能按 chunk_id 取回原文，缺失的 id 安静跳过；
- ``GET /kb/chunks/{id}`` 命中返回原文、未命中 404；
- ``GET /dept/gaps`` 与 ``/dept/gaps/download`` 的字段与文件名。
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import create_app
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.departments import DepartmentDirectory
from rag_assistant.insights import collect_gaps, gaps_markdown, load_records
from rag_assistant.kb import KnowledgeBase
from rag_assistant.vectorstore.base import VectorRecord
from rag_assistant.vectorstore.memory_store import MemoryVectorStore

REPO_DEPARTMENTS = Path(__file__).resolve().parents[1] / "config" / "departments.yaml"


def write_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _gap_records(tmp_path: Path, now: datetime):
    path = tmp_path / "logs" / "qa.jsonl"
    write_records(
        path,
        [
            {"ts": "2026-11-10T09:00:00", "question": "复学要交什么材料？", "refused": True, "departments": ["jwc"]},
            {"ts": "2026-11-10T09:05:00", "question": "复学要交什么材料", "refused": True, "departments": ["jwc"]},
            {"ts": "2026-11-10T09:06:00", "question": "复学需要带哪些材料", "refused": True, "departments": ["jwc"]},
            {"ts": "2026-11-09T09:00:00", "question": "图书馆能借几本书？", "refused": True, "departments": ["lib"]},
            {"ts": "2026-11-09T10:00:00", "question": "研究生推免名额怎么分配？", "refused": True, "departments": []},
            {"ts": "2026-11-09T11:00:00", "question": "休学最长可以多久", "refused": False, "departments": ["jwc"]},
        ],
    )
    return load_records(tmp_path / "logs", days=30, now=now)


def test_collect_gaps_groups_and_sorts(tmp_path: Path) -> None:
    now = datetime(2026, 11, 11, 9, 0, 0)
    records = _gap_records(tmp_path, now)
    report = collect_gaps(records, directory=DepartmentDirectory.load(REPO_DEPARTMENTS), days=30, now=now)

    assert report["total_refused"] == 5, "只统计被拒答的问题"
    assert report["unique_refused"] == 4, "标点差异算同一条问句（jwc 2 + lib 1 + 未识别 1）"
    departments = {row["id"]: row for row in report["departments"]}
    assert departments["jwc"]["count"] == 3
    assert departments["jwc"]["name"] == "教务处"
    assert departments["jwc"]["entries"] == 2, "两种说法算两个缺口"
    top = departments["jwc"]["items"][0]
    assert top["count"] == 2
    assert len(top["wordings"]) <= 3, "问法最多保留 3 条"
    assert report["unmatched_count"] == 1
    assert report["unmatched"][0]["question"] == "研究生推免名额怎么分配？"
    # 引用/部门排序：先按部门出现，未识别部门的放最后
    assert report["items"][-1]["dept_id"] == ""


def test_gaps_markdown_has_sections_and_hint() -> None:
    now = datetime(2026, 11, 11, 9, 0, 0)
    path_rows = [
        {"ts": "2026-11-10T09:00:00", "question": "复学要交什么材料？", "refused": True, "departments": ["jwc"]},
        {"ts": "2026-11-10T09:01:00", "question": "复学要交什么材料？", "refused": True, "departments": ["jwc"]},
        {"ts": "2026-11-10T09:02:00", "question": "天气怎么样", "refused": True, "departments": []},
    ]
    records = [{**row, "_ts": row["ts"], "_day": row["ts"][:10]} for row in path_rows]
    report = collect_gaps(records, directory=DepartmentDirectory.load(REPO_DEPARTMENTS), days=7, now=now)
    markdown = gaps_markdown(report, school_name="郑州大学", example_data=True)

    assert "制度文件补充清单" in markdown
    assert "教务处" in markdown
    assert "×2" in markdown
    assert "建议" in markdown
    assert "未识别到部门" in markdown
    assert "演示数据" in markdown


def test_gap_markdown_without_demo_notice() -> None:
    now = datetime(2026, 11, 11, 9, 0, 0)
    report = collect_gaps(
        [{"ts": "2026-11-10T09:00:00", "question": "复学材料", "refused": True, "departments": ["jwc"], "_ts": "2026-11-10T09:00:00", "_day": "2026-11-10"}],
        directory=DepartmentDirectory.load(REPO_DEPARTMENTS),
        days=7,
        now=now,
    )
    assert "演示数据" not in gaps_markdown(report)


# --------------------------------------------------------------------------
# 引用回查原文
# --------------------------------------------------------------------------
def _kb_with_chunks(mock_settings: Settings) -> tuple[KnowledgeBase, MemoryVectorStore]:
    kb = KnowledgeBase(mock_settings)
    store = MemoryVectorStore()
    store.upsert(
        [
            VectorRecord(
                chunk_id="c-1",
                text="休学期一般以一年为期，累计不得超过两年。",
                metadata={
                    "doc_id": "d-1",
                    "doc_name": "学生学籍管理规定.md",
                    "doc_type": "md",
                    "section_path": "第三章 > 第七条",
                    "page": 3,
                    "file_version": "abc123",
                },
                embedding=[1.0, 0.0],
            )
        ]
    )
    kb.vector_store = store  # type: ignore[assignment]
    return kb, store


def test_get_chunks_returns_original_text(mock_settings: Settings) -> None:
    kb, _store = _kb_with_chunks(mock_settings)
    chunks = kb.get_chunks(["c-1", "missing"])
    assert len(chunks) == 1, "不存在的 id 安静跳过"
    assert chunks[0]["text"].startswith("休学期一般以一年为期")
    assert chunks[0]["doc_name"] == "学生学籍管理规定.md"
    assert chunks[0]["section_path"] == "第三章 > 第七条"
    assert chunks[0]["page"] == 3


def test_chunks_endpoint_returns_404_for_unknown_id(mock_settings: Settings) -> None:
    kb, _store = _kb_with_chunks(mock_settings)
    app = create_app(mock_settings, chat_service=ChatService(mock_settings, kb=kb))
    with TestClient(app) as client:
        hit = client.get("/kb/chunks/c-1")
        assert hit.status_code == 200
        assert hit.json()["doc_name"] == "学生学籍管理规定.md"
        assert client.get("/kb/chunks/nope").status_code == 404


# --------------------------------------------------------------------------
# 看板接口：补料清单
# --------------------------------------------------------------------------
@pytest.fixture()
def gaps_client(mock_settings: Settings):
    mock_settings.log_dir.mkdir(parents=True, exist_ok=True)
    write_records(
        mock_settings.log_dir / "qa.jsonl",
        [
            {"ts": datetime.now().isoformat(timespec="seconds"), "question": "复学要交什么材料？", "refused": True, "departments": ["jwc"]},
            {"ts": datetime.now().isoformat(timespec="seconds"), "question": "图书馆能借几本书？", "refused": True, "departments": ["lib"]},
        ],
    )
    kb = KnowledgeBase(mock_settings)
    service = ChatService(mock_settings, kb=kb)
    app = create_app(mock_settings, chat_service=service)
    with TestClient(app) as client:
        yield client, mock_settings


def test_dept_gaps_endpoint(gaps_client) -> None:
    client, _settings = gaps_client
    payload = client.get("/dept/gaps?days=30").json()
    assert payload["total_refused"] == 2
    assert {row["id"] for row in payload["departments"]} == {"jwc", "lib"}
    assert payload["brand"]["school_name"] == "郑州大学"


def test_dept_gaps_download_uses_safe_filename(gaps_client) -> None:
    client, _settings = gaps_client
    response = client.get("/dept/gaps/download?days=7")
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert "filename*=UTF-8''" in disposition
    fallback = disposition.split("filename=")[1].split(";")[0].strip('"')
    fallback.encode("ascii")
    assert "复学要交什么材料" in response.text
