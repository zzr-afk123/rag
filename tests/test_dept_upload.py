"""部门看板：拖拽上传入库（``/dept/upload`` 及配套接口）。

覆盖：
- 正常上传 → 落盘 ``data/raw/uploads/`` → 触发增量同步 → manifest 记录入库；
- 同名冲突、显式覆盖；
- 路径穿越、非法扩展名、空文件、超大文件；
- 上传限流；
- 「最近上传」列表与页面上的拖拽上传入口。

为什么用原始字节体而不是 multipart：项目依赖里没有 ``python-multipart``，
上传接口刻意不引入新依赖（详见 ``api/routes/dept.py`` 的模块文档）。
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import create_app
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.kb import KnowledgeBase


def _build_client(settings: Settings) -> TestClient:
    kb = KnowledgeBase(settings)
    kb.sync()
    service = ChatService(settings, kb=kb)
    app = create_app(settings, chat_service=service)
    return TestClient(app)


@pytest.fixture()
def client(mock_settings: Settings) -> TestClient:
    with _build_client(mock_settings) as test_client:
        yield test_client


def _upload(client: TestClient, name: str, body: bytes, **params):
    return client.post("/dept/upload", content=body, params={"name": name, **params})


def _wait_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    """轮询同步任务直到结束（mock 链路通常 <1s，超时给足 CI 余量）。"""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        response = client.get(f"/dept/sync/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        status = (last.get("job") or {}).get("status")
        if status not in {"pending", "running"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"同步任务 {job_id} 超时未结束：{last}")


# --------------------------------------------------------------------------
# 正常路径
# --------------------------------------------------------------------------
def test_upload_lands_in_uploads_and_gets_indexed(client: TestClient, raw_dir: Path) -> None:
    body = "# 实验室安全管理办法\n\n第一条 进入实验室必须穿实验服并佩戴护目镜。\n".encode("utf-8")
    response = _upload(client, "实验室安全管理办法.md", body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["saved"] is True
    assert payload["name"] == "实验室安全管理办法.md"
    assert payload["rel_path"] == "uploads/实验室安全管理办法.md"
    assert payload["size"] == len(body)
    assert payload["overwritten"] is False

    saved = raw_dir / "uploads" / "实验室安全管理办法.md"
    assert saved.read_bytes() == body

    job_id = payload["sync"]["job_id"]
    assert job_id, payload
    result = _wait_job(client, job_id)
    assert result["job"]["status"] == "success"
    report = result["job"]["report"]
    assert report["added"] >= 1
    assert report["failed"] == []
    assert result["kb"]["documents"] >= 1

    listed = client.get("/dept/uploads").json()["files"]
    entry = next(item for item in listed if item["name"] == "实验室安全管理办法.md")
    assert entry["indexed"] is True
    assert entry["chunks"] > 0


def test_upload_is_reachable_from_chat_after_sync(client: TestClient) -> None:
    """闭环验证：上传的文档进入索引（清单/向量/BM25 三方一致）。"""
    body = (
        "# 实验动物管理办法\n\n"
        "第十二条 实验动物尸体必须交由后勤处统一无害化处理，严禁自行掩埋。\n"
    ).encode("utf-8")
    response = _upload(client, "实验动物管理办法.md", body)
    assert response.status_code == 200, response.text
    _wait_job(client, response.json()["sync"]["job_id"])

    status = client.get("/kb/status").json()
    assert status["documents"] >= 1
    assert status["vector_chunks"] == status["bm25_chunks"] > 0


# --------------------------------------------------------------------------
# 冲突与覆盖
# --------------------------------------------------------------------------
def test_upload_conflict_requires_overwrite(client: TestClient, raw_dir: Path) -> None:
    first = _upload(client, "冲突示例.md", "第一版".encode("utf-8"))
    assert first.status_code == 200, first.text
    _wait_job(client, first.json()["sync"]["job_id"])

    second = _upload(client, "冲突示例.md", "第二版".encode("utf-8"))
    assert second.status_code == 409
    assert "已存在同名文件" in second.json()["detail"]
    assert (raw_dir / "uploads" / "冲突示例.md").read_text(encoding="utf-8") == "第一版"

    third = _upload(client, "冲突示例.md", "第二版".encode("utf-8"), overwrite=True)
    assert third.status_code == 200, third.text
    assert third.json()["overwritten"] is True
    assert (raw_dir / "uploads" / "冲突示例.md").read_text(encoding="utf-8") == "第二版"
    _wait_job(client, third.json()["sync"]["job_id"])


# --------------------------------------------------------------------------
# 文件名校验
# --------------------------------------------------------------------------
def test_upload_strips_path_traversal(client: TestClient, project: Path, raw_dir: Path) -> None:
    response = _upload(client, "../../evil.md", "payload".encode("utf-8"))
    assert response.status_code == 200, response.text
    assert response.json()["rel_path"] == "uploads/evil.md"
    assert (raw_dir / "uploads" / "evil.md").exists()
    assert not (project / "evil.md").exists()
    _wait_job(client, response.json()["sync"]["job_id"])


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    response = _upload(client, "样本.exe", b"MZ")
    assert response.status_code == 415
    assert "不支持的文件类型" in response.json()["detail"]


def test_upload_rejects_empty_name(client: TestClient) -> None:
    response = _upload(client, "..", b"x")
    assert response.status_code == 400


def test_upload_rejects_empty_body(client: TestClient) -> None:
    response = _upload(client, "空文件.md", b"")
    assert response.status_code == 400
    assert "为空" in response.json()["detail"]


# --------------------------------------------------------------------------
# 体积与频率限制
# --------------------------------------------------------------------------
def test_upload_rejects_oversize(mock_settings: Settings) -> None:
    small = replace(
        mock_settings, dept_console=replace(mock_settings.dept_console, upload_max_mb=1)
    )
    with _build_client(small) as client:
        response = _upload(client, "big.txt", b"x" * (1024 * 1024 + 1))
        assert response.status_code == 413
        assert "上限 1 MB" in response.json()["detail"]


def test_upload_rate_limited(mock_settings: Settings) -> None:
    limited = replace(
        mock_settings, dept_console=replace(mock_settings.dept_console, upload_rate_per_min=1)
    )
    with _build_client(limited) as client:
        first = _upload(client, "限流一.md", "a".encode("utf-8"))
        assert first.status_code == 200, first.text
        _wait_job(client, first.json()["sync"]["job_id"])

        second = _upload(client, "限流二.md", "b".encode("utf-8"))
        assert second.status_code == 429
        assert "上传过于频繁" in second.json()["detail"]


# --------------------------------------------------------------------------
# 页面与列表接口
# --------------------------------------------------------------------------
def test_dept_page_has_upload_ui(client: TestClient) -> None:
    response = client.get("/dept")
    assert response.status_code == 200
    assert 'id="dropzone"' in response.text
    assert "上传制度文件" in response.text


def test_uploads_list_empty_initially(client: TestClient) -> None:
    payload = client.get("/dept/uploads").json()
    assert payload["files"] == []
    assert payload["upload_dir"].endswith("uploads")


def test_sync_job_unknown_id_returns_404(client: TestClient) -> None:
    assert client.get("/dept/sync/not-a-real-job").status_code == 404
