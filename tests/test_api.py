"""API 层测试：健康检查、状态、同步问答、SSE 流式、参数校验、错误映射。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import create_app
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.kb import KnowledgeBase


@pytest.fixture()
def client(mock_settings: Settings, sample_raw_dir: Path):
    """建好索引的测试客户端（mock 链路）。"""
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    service = ChatService(mock_settings, kb=kb)
    app = create_app(mock_settings, chat_service=service)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# 健康检查与状态
# --------------------------------------------------------------------------
def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["llm_mock"] is True
    assert payload["embedder_mock"] is True
    assert payload["chunks"] > 0
    assert payload["version"]


def test_kb_status(client: TestClient) -> None:
    response = client.get("/kb/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"] >= 4
    assert payload["vector_chunks"] == payload["bm25_chunks"] > 0
    assert payload["collection"].startswith("kb__")
    assert payload["syncing"] is False


def test_index_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # 公网首页（面向访客）
    assert "制度与办事流程问答" in response.text


def test_console_page(client: TestClient) -> None:
    response = client.get("/console")
    assert response.status_code == 200
    # 现在是运维台：顶部标题 + 实时数据接口
    assert "运维台" in response.text
    assert "/console/live" in response.text


# --------------------------------------------------------------------------
# 同步问答
# --------------------------------------------------------------------------
def test_chat_ok(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"session_id": "api-1", "question": "在读证明在哪里办？", "threshold": 0.0}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "api-1"
    assert payload["refused"] is False
    assert payload["citations"], "必须带引用"
    assert payload["citations"][0]["doc_name"] == "办事指南-在读证明.txt"
    assert payload["citations"][0]["quote"]
    assert payload["latency_ms"] >= 0
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["rewritten_query"]


def test_chat_refusal(client: TestClient) -> None:
    response = client.post(
        "/chat", json={"session_id": "api-2", "question": "图书馆几点开门？", "threshold": 0.5}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is True
    assert payload["citations"] == []
    assert payload["refusal_reason"]


def test_threshold_override_is_admin_only() -> None:
    """客户端不能把拒答闸门关掉——"关闸门"那一项消融的可信度前提在这里。

    以前 ``threshold`` 是请求体字段，任何登录者传 0 就能让弱资料成答；现在只有管理员
    能覆盖，其余角色返回 ``None`` = 用服务端 ``retrieval.score_threshold``。
    端到端的角色行为见 ``tests/test_auth.py::test_normal_user_cannot_move_refusal_gate``。
    """
    from types import SimpleNamespace

    from rag_assistant.api.routes.chat import _threshold

    def _request(role: str | None):
        user = None if role is None else SimpleNamespace(role=role)
        return SimpleNamespace(state=SimpleNamespace(user=user))

    def _body(value):
        return SimpleNamespace(threshold=value)

    assert _threshold(_request("admin"), _body(0.9)) == 0.9
    assert _threshold(_request("admin"), _body(None)) is None  # 没传也交给服务端配置
    for role in (None, "guest", "student", "dept"):
        assert _threshold(_request(role), _body(0.9)) is None, role


def test_chat_validation(client: TestClient) -> None:
    """空问题 / 超长问题 / 非法阈值都应由 pydantic 拦下（422）。"""
    assert client.post("/chat", json={"session_id": "s", "question": ""}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "question": "问" * 501}).status_code == 422
    assert (
        client.post("/chat", json={"session_id": "s", "question": "有效问题", "threshold": 2}).status_code
        == 422
    )
    assert client.post("/chat", json={"question": "缺少 session"}).status_code == 422


def test_session_isolation_via_api(client: TestClient) -> None:
    client.post("/chat", json={"session_id": "a", "question": "休学最长可以多久？", "threshold": 0.0})
    second = client.post(
        "/chat", json={"session_id": "b", "question": "在读证明怎么办？", "threshold": 0.0}
    ).json()
    assert "休学" not in second["rewritten_query"]


# --------------------------------------------------------------------------
# SSE 流式
# --------------------------------------------------------------------------
def _parse_sse(text: str) -> list[dict]:
    """把 SSE 响应体解析成事件列表。"""
    events: list[dict] = []
    for frame in text.split("\n\n"):
        data_lines = [line[5:].strip() for line in frame.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            events.append({"type": "DONE"})
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:  # pragma: no cover
            continue
    return events


def test_chat_stream_sse(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": "api-stream", "question": "在读证明怎么开？", "threshold": 0.0},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "event: meta" in body
    assert "event: token" in body
    assert "[DONE]" in body

    events = _parse_sse(body)
    kinds = [event["type"] for event in events]
    assert kinds[0] == "meta"
    assert "token" in kinds
    assert kinds[-2] == "done" or kinds[-1] == "done"

    done = [event for event in events if event["type"] == "done"][0]
    assert done["citations"]
    assert "引用来源" in done["answer"]


def test_chat_stream_refusal(client: TestClient) -> None:
    with client.stream(
        "POST", "/chat/stream", json={"session_id": "api-s2", "question": "食堂几点开门？", "threshold": 0.9}
    ) as response:
        body = "".join(response.iter_text())

    events = _parse_sse(body)
    done = [event for event in events if event["type"] == "done"][0]
    assert done["refused"] is True


def test_chat_stream_records_session(client: TestClient) -> None:
    with client.stream(
        "POST", "/chat/stream", json={"session_id": "api-s3", "question": "在读证明怎么开？", "threshold": 0.0}
    ) as response:
        "".join(response.iter_text())

    # 关闭登录时本地身份是 "local"；会话按 owner 加了一层前缀
    state = client.app.state.chat.sessions.list_turns("local::api-s3")
    assert len(state) == 1


# --------------------------------------------------------------------------
# 同步
# --------------------------------------------------------------------------
def test_kb_sync_wait_true(client: TestClient) -> None:
    """wait=true：同步执行并返回完整报告。"""
    response = client.post("/kb/sync", json={"rebuild": False, "wait": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["status"] in {"success", "failed"}
    assert payload["report"]["added"] == 0  # 已经同步过，这次应当是空操作
    assert payload["report"]["unchanged"] >= 4


def test_kb_sync_async(client: TestClient) -> None:
    """默认异步：202 + job_id，随后 /kb/status 的 syncing 会回落。"""
    response = client.post("/kb/sync", json={"rebuild": False})
    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get("/kb/status").json()
        if not status["syncing"]:
            break
        time.sleep(0.05)
    final = client.get("/kb/status").json()
    assert final["syncing"] is False
    assert final["last_report"] is not None


def test_kb_sync_rebuild(client: TestClient) -> None:
    response = client.post("/kb/sync", json={"rebuild": True, "wait": True})
    assert response.status_code == 200
    payload = response.json()
    assert payload["report"]["rebuild"] is True
    assert payload["report"]["added"] >= 4
