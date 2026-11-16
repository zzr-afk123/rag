"""登录相关接口：登录页、登录、访客、退出、当前用户。

Cookie 里只放随机会话令牌，用户信息在服务端（SessionStore）。
登录失败统一返回「编号或密码错误」，不区分"账号不存在"与"密码错误"。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from ...auth import UserStore, current_user, guest_user
from ...config import Settings
from ...logging_setup import get_logger
from ..security import client_ip, is_https

router = APIRouter(tags=["auth"])
logger = get_logger("api.auth")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class LoginBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=32, description="学号 / 工号")
    password: str = Field(..., min_length=1, max_length=128)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _users(request: Request) -> UserStore:
    return request.app.state.users


def _sessions(request: Request):
    return request.app.state.sessions


def _client_ip(request: Request) -> str:
    """登录节流的 key：默认用真实连接地址，只有配置信任代理时才看 X-Forwarded-For。"""
    settings = _settings(request)
    return client_ip(request, trust_forwarded_for=bool(settings.server.trust_forwarded_for))


def _grant(request: Request, response: Response, user) -> None:
    settings = _settings(request)
    token = _sessions(request).create(user)
    response.set_cookie(
        settings.auth.cookie_name,
        token,
        max_age=int(settings.auth.session_ttl_hours * 3600),
        httponly=True,
        samesite="lax",
        # 加密连接自动加 Secure；显式开关用于"就算代理没改写协议也必须 Secure"的部署。
        # 明文 HTTP 下强制 Secure 会让任何人都登不上，所以不是无条件 true。
        secure=bool(settings.auth.cookie_secure) or is_https(request),
        path="/",
    )


@router.get("/login", include_in_schema=False, response_model=None)
def login_page(request: Request):
    settings = _settings(request)
    if not settings.auth.enabled:
        return RedirectResponse("/", status_code=302)
    page = WEB_DIR / "login.html"
    if not page.exists():  # pragma: no cover
        return JSONResponse({"detail": "登录页缺失（web/login.html）"}, status_code=404)
    return FileResponse(page, headers={"Cache-Control": "no-store"})


@router.post("/auth/login", summary="登录（编号 + 密码）")
def login(body: LoginBody, request: Request, response: Response):
    settings = _settings(request)
    if not settings.auth.enabled:
        return {"ok": True, "user": {"id": "local", "name": "本机", "role": "admin", "role_label": "管理员"}}

    throttle = getattr(request.app.state, "login_throttle", None)
    if throttle is not None and not throttle.allow(_client_ip(request)):
        return JSONResponse({"detail": "登录尝试过于频繁，请 1 分钟后再试"}, status_code=429)

    user = _users(request).authenticate(body.id.strip(), body.password)
    if user is None:
        logger.info("登录失败：id=%r ip=%s", body.id, _client_ip(request))
        return JSONResponse({"detail": "编号或密码错误，请检查后重试"}, status_code=401)

    _grant(request, response, user)
    logger.info("登录成功：%s（%s）ip=%s", user.name, user.role_label, _client_ip(request))
    return {"ok": True, "user": user.public()}


@router.post("/auth/guest", summary="访客进入（只读问答）")
def guest(request: Request, response: Response):
    settings = _settings(request)
    if not settings.auth.enabled:
        return {"ok": True, "user": {"id": "local", "name": "本机", "role": "admin", "role_label": "管理员"}}
    if not settings.auth.allow_guest:
        return JSONResponse({"detail": "访客入口已关闭，请使用编号登录"}, status_code=403)
    user = guest_user()
    _grant(request, response, user)
    logger.info("访客进入：%s ip=%s", user.id, _client_ip(request))
    return {"ok": True, "user": user.public()}


@router.post("/auth/logout", summary="退出登录")
def logout(request: Request, response: Response):
    sessions = _sessions(request)
    token = request.cookies.get(_settings(request).auth.cookie_name)
    sessions.drop(token)
    response.delete_cookie(_settings(request).auth.cookie_name, path="/")
    return {"ok": True}


@router.get("/auth/me", summary="当前登录身份")
def me(request: Request):
    user = current_user(request)
    if user is None:
        return JSONResponse({"detail": "未登录"}, status_code=401)
    return {"user": user.public()}
