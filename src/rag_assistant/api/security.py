"""公网暴露的最低限度保护：访问令牌 + 每 IP 限流。

为什么需要（需求：要能从互联网访问）：
- 一旦挂到公网，`/chat` 就是**别人拿你的额度去刷 LLM** 的入口；
- `/kb/sync` 会重写索引，被反复触发等于 DoS。

这里只做两件很小的事，够用就好，不做用户体系：
1. **访问令牌**（可选）：非空时，`/chat*` 与 `/kb/*` 必须带
   ``X-Access-Token`` 头或 ``?token=`` 查询参数。给朋友用就发带 token 的链接。
2. **每 IP 限流**（默认 60 次/分钟，可关）：滑动窗口，超限返回 429 + Retry-After。

刻意不做：登录态、按用户配额、验证码。这些交给网关（Nginx/Caddy/Cloudflare）更合适。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

#: 需要保护的前缀（只读接口 /health、静态页不限制，便于探活与打开页面）
PROTECTED_PREFIXES = ("/chat", "/kb/")

#: 部门看板的数据接口（不含 `/dept` 页面本身：页面要能被打开并让用户填令牌）
DEPT_PREFIXES = ("/dept/",)

#: 限流表里最多保留多少个 key；超过就丢掉已经空掉的桶，防止被随机 IP 撑爆内存
MAX_RATE_KEYS = 4096


def client_ip(request: Request, *, trust_forwarded_for: bool = False) -> str:
    """取调用方 IP。

    ``X-Forwarded-For`` 是客户端可以随便写的，只有在明确知道"自己一定在反向代理后面"
    （``server.trust_forwarded_for: true``）时才采信第一段；否则一律用 TCP 连接地址，
    不然刷口令的人每次换一个假 IP 就能绕过登录节流与限流。
    """
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def is_https(request: Request) -> bool:
    """这次请求是不是加密进来的。

    只看 ``scope["scheme"]``：uvicorn 在 ``proxy_headers`` 打开且连接方在
    ``forwarded_allow_ips`` 名单里时，才会用 ``X-Forwarded-Proto`` 改写它。
    自己再去读一遍原始头反而危险——明文 HTTP 下客户端可以伪造这个头。
    """
    return request.url.scheme == "https"


class RateLimiter:
    """按 key（通常是 IP）的滑动窗口计数器。进程内实现，单 worker 假设。"""

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0) -> None:
        self.limit = limit_per_minute
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """返回 (是否放行, 需等待秒数)。"""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > MAX_RATE_KEYS:
                self._sweep(now)
            bucket = self._hits[key]
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry = int(self.window - (now - bucket[0])) + 1
                return False, retry
            bucket.append(now)
        return True, 0

    def _sweep(self, now: float) -> None:
        """丢掉窗口已过期的 key，避免字典无限增长（调用方需持锁）。"""
        dead = [key for key, bucket in self._hits.items() if not bucket or now - bucket[-1] > self.window]
        for key in dead:
            self._hits.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


class AccessControlMiddleware(BaseHTTPMiddleware):
    """令牌校验 + 限流。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str = "",
        rate_limit_per_min: int = 60,
        dept_token: str = "",
        dept_prefixes: tuple[str, ...] = DEPT_PREFIXES,
        trust_forwarded_for: bool = False,
    ) -> None:
        super().__init__(app)
        self._token = token.strip()
        self._dept_token = dept_token.strip()
        self._dept_prefixes = dept_prefixes
        self._trust_forwarded_for = trust_forwarded_for
        self._limiter = RateLimiter(rate_limit_per_min)

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN201
        path = request.url.path
        protected = path.startswith(PROTECTED_PREFIXES)
        dept_path = path.startswith(self._dept_prefixes)
        if not protected and not dept_path:
            return await call_next(request)

        required = self._token if protected else self._dept_token
        if required:
            provided = request.headers.get("x-access-token") or request.query_params.get("token")
            if provided != required:
                return JSONResponse(
                    {
                        "detail": "需要访问令牌：请使用带 ?token=… 的链接，或设置 X-Access-Token 头"
                        if protected
                        else "部门看板需要访问令牌：请用带 ?token=… 的链接打开，或设置 X-Access-Token 头"
                    },
                    status_code=401,
                )

        # 部门看板是只读统计，不占问答限流额度（否则看板轮询会把用户挤掉）
        if dept_path:
            return await call_next(request)

        client_ip_value = client_ip(request, trust_forwarded_for=self._trust_forwarded_for)
        allowed, retry_after = self._limiter.check(client_ip_value)
        if not allowed:
            return JSONResponse(
                {"detail": f"请求过于频繁（{self._limiter.limit}/分钟），请 {retry_after}s 后再试"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
