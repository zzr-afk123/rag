"""身份与权限：编号/密码登录、访客、按角色控制可见范围。

三个角色，一条规则：

- ``student``（学生）：只能用问答页；看不到部门看板、周报与运维台。
- ``dept``（部门，含 ``admin``）：问答页 + 部门看板 + 运维台，全部可见。
- ``guest``（访客）：只能问答，且不带部门入口。

**服务端强制，前端只做"藏入口"**：即使学生手敲 ``/dept`` 或直接请求
``/dept/summary``，也会被中间件挡住（页面 403 / 接口 403 JSON），而不是靠隐藏链接。

密码存储：PBKDF2-HMAC-SHA256（12 万次迭代 + 16 字节随机盐），格式
``pbkdf2_sha256$迭代次数$盐$哈希``，用 ``hmac.compare_digest`` 比对。
YAML 里永远只有哈希，不存明文；初始账号由 ``scripts/manage_users.py`` 生成。

会话：进程内字典（单 worker 假设），带 TTL；Cookie 为 HttpOnly + SameSite=Lax，
HTTPS 下把 ``auth.cookie_secure`` 设为 true。重启服务需要重新登录——这是刻意的
取舍：不引入 Redis/数据库，换来的是一次重启后的干净状态。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .logging_setup import get_logger

logger = get_logger("auth")

PBKDF2_PREFIX = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 120_000

ROLE_LABELS = {"student": "学生", "dept": "部门", "admin": "管理员", "guest": "访客"}
DEPT_ROLES = {"dept", "admin"}

#: `scripts/manage_users.py init` 生成的示例口令；启动时若仍有账号在用，打警告
_SEED_PASSWORDS: tuple[str, ...] = ("student123", "dept123", "admin123", "123456", "password")

#: 不需要登录的路径（登录页、静态资源、探活、品牌信息）
#: 注意 /auth/me 必须经过中间件（要把用户写进 request.state），不能整个 /auth/ 放行
OPEN_PREFIXES = ("/login", "/static/", "/health", "/brand")
#: 这几个接口用于建立/结束会话，未登录也能访问
OPEN_EXACT = {"/auth/login", "/auth/guest", "/auth/logout"}
#: 只有部门/管理员能看的路径（页面与接口）
DEPT_PREFIXES = ("/dept", "/console")
# 同步接口会写索引，只有部门能触发；/kb/status 只是只读计数，登录后都能看
DEPT_EXACT = {"/kb/sync"}
#: 这些前缀未登录时返回 401 JSON，而不是跳转登录页
API_PREFIXES = ("/chat", "/kb/", "/dept/", "/auth/", "/brand")


# --------------------------------------------------------------------------
# 密码
# --------------------------------------------------------------------------
def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS, salt: bytes | None = None) -> str:
    """生成 ``pbkdf2_sha256$迭代$盐$哈希`` 字符串。"""
    if not password:
        raise ValueError("密码不能为空")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "$".join(
        (
            PBKDF2_PREFIX,
            str(iterations),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, stored: str) -> bool:
    """校验密码；格式不认识或字段缺失一律判失败（不抛异常）。"""
    if not password or not stored:
        return False
    try:
        prefix, iterations, salt_b64, digest_b64 = stored.split("$")
        if prefix != PBKDF2_PREFIX:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


# --------------------------------------------------------------------------
# 用户
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: str = "student"
    department: str = ""
    active: bool = True
    password_hash: str = field(default="", repr=False)

    @property
    def is_dept(self) -> bool:
        return self.role in DEPT_ROLES

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "role_label": self.role_label,
            "department": self.department,
            "can_view_dept": self.is_dept,
        }


class UserStore:
    """从 YAML 读取用户；文件变化时自动重载（改密不用重启）。"""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._users: dict[str, User] = {}
        self._mtime: float = 0.0
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._users = {}
            if not self.path or not self.path.exists():
                logger.warning("用户库不存在：%s（登录将不可用）", self.path)
                return
            try:
                payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # noqa: BLE001 - 配置坏了不该让服务起不来
                logger.error("用户库解析失败：%s（%s）", self.path, exc)
                return
            entries = payload.get("users") if isinstance(payload, dict) else payload
            if not isinstance(entries, list):
                logger.error("用户库格式应为 users: [...]: %s", self.path)
                return
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                user_id = str(entry.get("id") or "").strip()
                if not user_id:
                    continue
                role = str(entry.get("role") or "student").strip().lower()
                if role not in ROLE_LABELS:
                    logger.warning("未知角色 %r（用户 %s），按 student 处理", role, user_id)
                    role = "student"
                self._users[user_id] = User(
                    id=user_id,
                    name=str(entry.get("name") or user_id),
                    role=role,
                    department=str(entry.get("department") or ""),
                    active=bool(entry.get("active", True)),
                    password_hash=str(entry.get("password") or ""),
                )
            try:
                self._mtime = self.path.stat().st_mtime
            except OSError:
                self._mtime = 0.0
            logger.info("用户库已加载：%d 个账号（%s）", len(self._users), self.path)

    def _maybe_reload(self) -> None:
        if not self.path:
            return
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            logger.info("检测到用户库变化，重新加载")
            self.reload()

    def get(self, user_id: str) -> User | None:
        self._maybe_reload()
        return self._users.get(user_id)

    def authenticate(self, user_id: str, password: str) -> User | None:
        user = self.get(user_id)
        if user is None or not user.active:
            return None
        return user if verify_password(password, user.password_hash) else None

    def all(self) -> Iterable[User]:
        self._maybe_reload()
        return tuple(self._users.values())

    def weak_accounts(self, candidates: Sequence[str] = _SEED_PASSWORDS) -> list[str]:
        """返回仍在使用示例口令的账号 id（启动时告警，提示尽快改密码）。

        只做一次 PBKDF2 校验，账号数量是个位数，启动多花几百毫秒可以接受。
        """
        hits: list[str] = []
        for user in self.all():
            if not user.active:
                continue
            for candidate in candidates:
                if verify_password(candidate, user.password_hash):
                    hits.append(user.id)
                    break
        return hits


# --------------------------------------------------------------------------
# 会话
# --------------------------------------------------------------------------
class SessionStore:
    """进程内会话表（token -> 用户 + 过期时间）。"""

    def __init__(self, ttl_seconds: int = 12 * 3600) -> None:
        self.ttl = max(60, int(ttl_seconds))
        self._data: dict[str, tuple[User, float]] = {}
        self._lock = threading.Lock()

    def create(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._data[token] = (user, time.time() + self.ttl)
        return token

    def get(self, token: str | None) -> User | None:
        if not token:
            return None
        now = time.time()
        with self._lock:
            item = self._data.get(token)
            if item is None:
                return None
            user, expires = item
            if expires < now:
                self._data.pop(token, None)
                return None
            return user

    def drop(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._data.pop(token, None)

    def cleanup(self) -> int:
        now = time.time()
        with self._lock:
            dead = [key for key, (_, expires) in self._data.items() if expires < now]
            for key in dead:
                self._data.pop(key, None)
        return len(dead)

    @property
    def size(self) -> int:
        return len(self._data)


def guest_user() -> User:
    """访客：随机 id，不落库，只能问答。"""
    return User(id="guest-" + secrets.token_hex(3), name="访客", role="guest")


# --------------------------------------------------------------------------
# 登录限流（防暴力猜密码）
# --------------------------------------------------------------------------
class LoginThrottle:
    def __init__(self, limit_per_minute: int = 10) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.time()
        with self._lock:
            bucket = [stamp for stamp in self._hits.get(key, []) if now - stamp < 60]
            if len(bucket) >= self.limit:
                self._hits[key] = bucket
                return False
            bucket.append(now)
            self._hits[key] = bucket
        return True


# --------------------------------------------------------------------------
# 中间件
# --------------------------------------------------------------------------
def _wants_json(request: Request) -> bool:
    path = request.url.path
    if path.startswith(API_PREFIXES):
        return True
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


def _forbidden_page(message: str) -> HTMLResponse:
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>无访问权限</title>
<style>
  body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#f4f7fb; color:#12233c;
         font:15px/1.7 "PingFang SC","Microsoft YaHei",sans-serif; }}
  .card {{ background:#fff; border:1px solid #dde5f0; border-radius:16px; padding:32px 36px;
          box-shadow:0 12px 32px rgba(13,76,146,.08); max-width:420px; text-align:center; }}
  h1 {{ font-size:19px; margin:0 0 8px; color:#0d4c92; }}
  p {{ color:#5c6f88; font-size:13.5px; margin:0 0 18px; }}
  a {{ display:inline-block; background:#0d4c92; color:#fff; text-decoration:none;
       padding:9px 20px; border-radius:10px; font-size:14px; }}
</style></head>
<body><div class="card">
  <h1>无访问权限</h1>
  <p>{message}</p>
  <a href="/">返回问答页</a>
</div></body></html>"""
    return HTMLResponse(html, status_code=403)


class AuthMiddleware(BaseHTTPMiddleware):
    """按角色控制页面与接口；未登录时页面跳登录、接口返回 401。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        users: UserStore,
        sessions: SessionStore,
        enabled: bool = True,
        cookie_name: str = "rag_session",
    ) -> None:
        super().__init__(app)
        self._users = users
        self._sessions = sessions
        self._enabled = enabled
        self._cookie = cookie_name

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001, ANN201
        if not self._enabled:
            # 未启用鉴权时，视为本机管理员，便于开发与自动化测试
            request.state.user = User(id="local", name="本机", role="admin")
            return await call_next(request)

        path = request.url.path
        if path in OPEN_EXACT or path.startswith(OPEN_PREFIXES):
            return await call_next(request)

        user = self._sessions.get(request.cookies.get(self._cookie))
        request.state.user = user

        if user is None:
            if _wants_json(request):
                return JSONResponse({"detail": "未登录或登录已过期，请重新登录"}, status_code=401)
            target = path + (("?" + request.url.query) if request.url.query else "")
            return RedirectResponse(f"/login?next={target}", status_code=302)

        dept_only = path.startswith(DEPT_PREFIXES) or path in DEPT_EXACT
        if dept_only and not user.is_dept:
            message = f"当前身份为「{user.role_label}」，部门看板与运维台仅对部门账号开放。"
            if _wants_json(request):
                return JSONResponse({"detail": message}, status_code=403)
            return _forbidden_page(message)

        return await call_next(request)


def current_user(request: Request) -> User | None:
    """路由里取当前用户（中间件已写入 state）。"""
    return getattr(request.state, "user", None)
