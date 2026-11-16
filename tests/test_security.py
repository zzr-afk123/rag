"""公网访问保护测试：令牌校验 + 限流。"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_assistant.api.security import AccessControlMiddleware, RateLimiter


def _client(token: str = "", rate: int = 60) -> TestClient:
    app = FastAPI()
    app.add_middleware(AccessControlMiddleware, token=token, rate_limit_per_min=rate)

    @app.post("/chat")
    def chat() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/kb/status")
    def status() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/")
    def index() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_public_endpoints_stay_open() -> None:
    """静态页与 /health 不能被令牌挡住，否则探活和打开页面都会失败。"""
    client = _client(token="secret")
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_token_required_for_chat() -> None:
    client = _client(token="secret")
    assert client.post("/chat").status_code == 401
    assert client.post("/chat", headers={"X-Access-Token": "secret"}).status_code == 200
    assert client.post("/chat?token=secret").status_code == 200


def test_wrong_token_rejected() -> None:
    client = _client(token="secret")
    assert client.post("/chat", headers={"X-Access-Token": "nope"}).status_code == 401


def test_no_token_configured_allows_all() -> None:
    client = _client(token="")
    assert client.post("/chat").status_code == 200


def test_rate_limit_returns_429() -> None:
    client = _client(token="", rate=2)
    assert client.post("/chat").status_code == 200
    assert client.post("/chat").status_code == 200
    blocked = client.post("/chat")
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers


def test_rate_limiter_window_slides() -> None:
    limiter = RateLimiter(2, window_seconds=0.05)
    assert limiter.check("ip")[0] is True
    assert limiter.check("ip")[0] is True
    assert limiter.check("ip")[0] is False
    # 等窗口滑过后应恢复。睡眠要留足余量：Windows 上 monotonic 的分辨率约
    # 15.6ms，只睡 60ms 时会量出 46ms 左右，仍落在 50ms 窗口内，测试随机失败。
    import time

    time.sleep(0.2)
    assert limiter.check("ip")[0] is True


def test_rate_limiter_zero_means_unlimited() -> None:
    limiter = RateLimiter(0)
    for _ in range(100):
        assert limiter.check("ip")[0] is True


def _spoof_client(trust: bool, rate: int = 1) -> TestClient:
    """rate=1：同一个真实 IP 第二次请求必被限流。"""
    app = FastAPI()
    app.add_middleware(
        AccessControlMiddleware, token="", rate_limit_per_min=rate, trust_forwarded_for=trust
    )

    @app.post("/chat")
    def chat() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_forwarded_for_ignored_by_default() -> None:
    """默认不信任 X-Forwarded-For：伪造不同的头也必须被同一条限流记录挡住。"""
    client = _spoof_client(trust=False, rate=1)
    assert client.post("/chat", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    second = client.post("/chat", headers={"X-Forwarded-For": "2.2.2.2"})
    assert second.status_code == 429


def test_forwarded_for_used_when_trusted() -> None:
    """显式开启后，不同来源 IP 各自计数（反向代理场景）。"""
    client = _spoof_client(trust=True, rate=1)
    assert client.post("/chat", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert client.post("/chat", headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    assert client.post("/chat", headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429


def test_rate_limiter_sweeps_stale_keys(monkeypatch) -> None:
    """大量一次性来源不能把内存无限撑大（阈值调小以便触发）。"""
    import time

    from rag_assistant.api import security

    monkeypatch.setattr(security, "MAX_RATE_KEYS", 10)
    limiter = RateLimiter(5, window_seconds=0.05)
    for i in range(12):
        limiter.check(f"ip-{i}")
    assert len(limiter._hits) == 12
    time.sleep(0.15)
    limiter.check("fresh")  # 超过阈值，触发清理
    assert len(limiter._hits) < 12
