"""登录与角色权限测试：密码哈希、会话、页面/接口的 401/403、访客与退出。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_assistant.api.main import create_app
from rag_assistant.auth import (
    SessionStore,
    User,
    UserStore,
    guest_user,
    hash_password,
    verify_password,
)
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.kb import KnowledgeBase

USERS_YAML = """users:
  - id: "20230001"
    name: 张三
    role: student
    department: ""
    password: "{student_hash}"
    active: true
  - id: jwc01
    name: 王老师
    role: dept
    department: 教务处
    password: "{dept_hash}"
    active: true
  - id: gone
    name: 停用账号
    role: dept
    department: 教务处
    password: "{dept_hash}"
    active: false
"""


# --------------------------------------------------------------------------
# 密码与会话
# --------------------------------------------------------------------------
def test_password_hash_roundtrip() -> None:
    stored = hash_password("student123")
    assert stored.startswith("pbkdf2_sha256$")
    assert verify_password("student123", stored) is True
    assert verify_password("wrong", stored) is False
    assert verify_password("", stored) is False
    assert verify_password("student123", "plain-text-not-hash") is False


def test_same_password_gets_different_salt() -> None:
    assert hash_password("x") != hash_password("x")


def test_user_store_and_authenticate(tmp_path: Path) -> None:
    path = tmp_path / "users.yaml"
    path.write_text(
        USERS_YAML.format(student_hash=hash_password("s1"), dept_hash=hash_password("d1")),
        encoding="utf-8",
    )
    store = UserStore(path)
    user = store.authenticate("20230001", "s1")
    assert user is not None and user.role == "student" and not user.is_dept
    dept = store.authenticate("jwc01", "d1")
    assert dept is not None and dept.is_dept
    assert store.authenticate("20230001", "d1") is None
    assert store.authenticate("nobody", "s1") is None
    assert store.authenticate("gone", "d1") is None, "停用账号不能登录"


def test_session_ttl_and_drop() -> None:
    sessions = SessionStore(ttl_seconds=60)
    token = sessions.create(User(id="u1", name="张三"))
    assert sessions.get(token) is not None
    sessions.drop(token)
    assert sessions.get(token) is None
    assert sessions.get(None) is None
    assert sessions.get("forged-token") is None


def test_guest_user_is_not_dept() -> None:
    guest = guest_user()
    assert guest.role == "guest" and not guest.is_dept
    assert guest.public()["can_view_dept"] is False


# --------------------------------------------------------------------------
# HTTP：登录 / 权限
# --------------------------------------------------------------------------
@pytest.fixture()
def auth_settings(mock_settings: Settings, tmp_path: Path) -> Settings:
    path = tmp_path / "config" / "users.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        USERS_YAML.format(student_hash=hash_password("s1"), dept_hash=hash_password("d1")),
        encoding="utf-8",
    )
    return replace(
        mock_settings,
        auth=replace(
            mock_settings.auth,
            enabled=True,
            users_file=str(path),
            allow_guest=True,
            login_limit_per_min=5,
        ),
    )


@pytest.fixture()
def auth_client(auth_settings: Settings):
    (auth_settings.log_dir).mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(auth_settings)
    service = ChatService(auth_settings, kb=kb)
    app = create_app(auth_settings, chat_service=service)
    with TestClient(app) as client:
        yield client


def login(client: TestClient, user_id: str, password: str):
    return client.post("/auth/login", json={"id": user_id, "password": password})


def test_login_page_is_public(auth_client: TestClient) -> None:
    response = auth_client.get("/login")
    assert response.status_code == 200
    assert "访客进入" in response.text


def test_root_redirects_to_login_when_anonymous(auth_client: TestClient) -> None:
    response = auth_client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("/login?next=")


def test_chat_returns_401_json_when_anonymous(auth_client: TestClient) -> None:
    response = auth_client.post("/chat", json={"session_id": "s", "question": "休学多久"})
    assert response.status_code == 401
    assert "未登录" in response.json()["detail"]


def test_login_success_and_me(auth_client: TestClient) -> None:
    response = login(auth_client, "20230001", "s1")
    assert response.status_code == 200
    payload = response.json()["user"]
    assert payload["role"] == "student" and payload["can_view_dept"] is False
    me = auth_client.get("/auth/me").json()["user"]
    assert me["id"] == "20230001"


def test_login_wrong_password(auth_client: TestClient) -> None:
    response = login(auth_client, "20230001", "wrong")
    assert response.status_code == 401
    assert "编号或密码错误" in response.json()["detail"]


def test_student_cannot_open_dept_page_or_api(auth_client: TestClient) -> None:
    login(auth_client, "20230001", "s1")
    page = auth_client.get("/dept", follow_redirects=False)
    assert page.status_code == 403
    assert "仅对部门账号开放" in page.text

    api = auth_client.get("/dept/summary")
    assert api.status_code == 403

    # 只读计数学生也能看（前端用它判断知识库是否就绪）
    assert auth_client.get("/kb/status").status_code == 200
    # 触发同步会写索引，只有部门能点
    assert auth_client.post("/kb/sync", json={"rebuild": False}).status_code == 403


def test_dept_can_open_dept_page_and_api(auth_client: TestClient) -> None:
    login(auth_client, "jwc01", "d1")
    assert auth_client.get("/dept").status_code == 200
    summary = auth_client.get("/dept/summary")
    assert summary.status_code == 200
    assert summary.json()["total"] == 0


def test_guest_can_chat_page_but_not_dept(auth_client: TestClient) -> None:
    assert auth_client.post("/auth/guest").status_code == 200
    assert auth_client.get("/").status_code == 200
    assert auth_client.get("/dept").status_code == 403
    me = auth_client.get("/auth/me").json()["user"]
    assert me["role"] == "guest" and me["can_view_dept"] is False


def test_logout_drops_session(auth_client: TestClient) -> None:
    login(auth_client, "jwc01", "d1")
    assert auth_client.get("/auth/me").status_code == 200
    assert auth_client.post("/auth/logout").status_code == 200
    assert auth_client.get("/auth/me").status_code == 401
    assert auth_client.get("/dept", follow_redirects=False).status_code == 302


def test_login_rate_limit(auth_client: TestClient) -> None:
    for _ in range(5):
        login(auth_client, "20230001", "wrong")
    response = login(auth_client, "20230001", "wrong")
    assert response.status_code == 429


def test_student_cannot_download_dept_report(auth_client: TestClient) -> None:
    login(auth_client, "20230001", "s1")
    assert auth_client.get("/dept/report?dept=jwc&days=7").status_code == 403


def test_missing_users_file_refuses_to_start(mock_settings: Settings, tmp_path: Path) -> None:
    """开启登录但账号库是空的：必须启动失败，而不是悄悄变成"人人都是管理员"。"""
    from rag_assistant.exceptions import RagAssistantError

    settings = replace(
        mock_settings,
        auth=replace(
            mock_settings.auth,
            enabled=True,
            users_file=str(tmp_path / "config" / "nobody.yaml"),
        ),
    )
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RagAssistantError, match="拒绝启动"):
        create_app(settings)


def test_missing_users_file_can_be_allowed_explicitly(
    mock_settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    """本机演示允许用 RAG_ALLOW_NO_USERS=1 显式跳过，日志里会有警告。"""
    monkeypatch.setenv("RAG_ALLOW_NO_USERS", "1")
    settings = replace(
        mock_settings,
        auth=replace(
            mock_settings.auth,
            enabled=True,
            users_file=str(tmp_path / "config" / "nobody.yaml"),
        ),
    )
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
