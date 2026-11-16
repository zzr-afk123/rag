"""FastAPI 应用工厂与启动入口。

启动方式::

    uvicorn rag_assistant.api.main:app --host 0.0.0.0 --port 8000
    python -m rag_assistant.api.main
    rag-api                      # pip install -e . 后可用（见 pyproject.toml）

设计要点：
- **应用工厂** ``create_app()`` 支持注入 ``Settings`` / ``ChatService``，测试可以
  给每个用例一套独立临时目录；
- 重活（加载 BM25、连 Chroma、建会话存储）都放在 ``lifespan`` 里，
  保证 ``import`` 阶段不产生副作用，``--reload`` 也不会反复初始化；
- **单 worker**：向量库是嵌入式 Chroma、会话在进程内存、同步锁是进程内锁，
  多 worker 会各持一份索引。Docker 里固定 ``--workers 1``（README 已写明）。
"""

from __future__ import annotations

import mimetypes
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..auth import AuthMiddleware, LoginThrottle, SessionStore, UserStore
from ..chat_service import ChatService
from ..config import Settings, get_settings
from ..departments import DepartmentDirectory
from ..exceptions import ConfigError, IndexBusyError, LLMError, RagAssistantError
from ..kb import KnowledgeBase
from ..logging_setup import get_logger, setup_logging
from ..insights import WeeklyReportScheduler
from .jobs import SyncJobManager
from .routes import auth as auth_router
from .routes import brand as brand_router
from .routes import chat as chat_router
from .routes import console as console_router
from .routes import dept as dept_router
from .routes import health as health_router
from .routes import kb as kb_router
from .security import AccessControlMiddleware, RateLimiter
from ..sync.scheduler import SyncScheduler
from ..sync.watcher import RawDirWatcher

logger = get_logger("api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# .webmanifest 不在 Python 标准 mimetypes 里，不显式注册会以 text/plain 下发，
# 部分浏览器（尤其安卓 Chrome）会因此不认 manifest。
mimetypes.add_type("application/manifest+json", ".webmanifest")
DESCRIPTION = """
制度与办事流程 RAG 智能问答助手的 HTTP 接口。

- `POST /chat` 同步问答（含引用来源）
- `POST /chat/stream` SSE 流式问答
- `POST /kb/sync` 手动触发知识库同步（默认异步）
- `GET /kb/status` 索引状态
- `GET /health` 健康检查
- `GET /brand` 校名/校训/部门联系方式（公开）
- `GET /dept/summary` 部门看板统计（近 N 天问题分布）
- `GET /dept/report` 周报预览（`download=true` 导出 Excel）；`POST /dept/report/send` 立即投递

所有回答都基于知识库原文；证据不足时返回统一拒答话术并给出最相近文档线索。
"""


def create_app(
    settings: Settings | None = None,
    *,
    chat_service: ChatService | None = None,
    kb: KnowledgeBase | None = None,
) -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        settings: 配置；缺省读全局单例（.env + settings.yaml）。
        chat_service: 注入现成的对话服务（测试用）。
        kb: 注入现成的知识库（测试用）；``chat_service`` 优先。
    """
    cfg = settings or get_settings()

    # 登录与角色：账号在 config/users.yaml（只存密码哈希），会话在进程内
    users = UserStore(cfg.users_file)
    sessions = SessionStore(int(cfg.auth.session_ttl_hours * 3600))
    login_throttle = LoginThrottle(cfg.auth.login_limit_per_min)
    auth_enabled = bool(cfg.auth.enabled and len(list(users.all())) > 0)
    if cfg.auth.enabled and not auth_enabled:
        hint = "请先运行 scripts/manage_users.py init 建账号。"
        if os.environ.get("RAG_ALLOW_NO_USERS", "").strip() == "1":
            logger.warning(
                "auth.enabled=true 但用户库为空，已按 RAG_ALLOW_NO_USERS=1 关闭登录校验"
                "（仅限本机演示，不要用于公网）。%s",
                hint,
            )
        else:
            raise ConfigError(
                "auth.enabled=true 但用户库为空，拒绝启动：否则任何人都能以访客身份进来。"
                + hint
                + "本机演示可设 RAG_ALLOW_NO_USERS=1 跳过。"
            )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(cfg)
        service = chat_service or ChatService(cfg, kb=kb)
        jobs = SyncJobManager(service.kb)

        def _on_dir_change(reason: str) -> None:
            """watchdog 回调：与手动同步共用同一套单飞任务管理。"""
            try:
                jobs.trigger(rebuild=False)
            except IndexBusyError:
                logger.info("目录变更触发同步被跳过（已有任务在跑）：%s", reason)
            except Exception:  # noqa: BLE001 - 回调不能抛到监听线程
                logger.exception("目录变更触发同步失败：%s", reason)

        scheduler = SyncScheduler(cfg, jobs)
        scheduler.start()
        watcher = RawDirWatcher(cfg, _on_dir_change)
        watcher.start()
        # 部门目录只加载一次：问答（提示联系方式）与看板（统计归属）共用同一份
        departments = DepartmentDirectory.from_settings(cfg)
        weekly = WeeklyReportScheduler(cfg)
        weekly.start()

        app.state.settings = cfg
        app.state.chat = service
        # 运维台展示"已经运行多久"，用进程内的启动时刻算
        app.state.started_at = time.time()
        app.state.jobs = jobs
        # 部门看板的拖拽上传是写操作：按账号（未登录时按 IP）限流，防止被当成文件服务器刷
        app.state.upload_limiter = RateLimiter(cfg.dept_console.upload_rate_per_min)
        app.state.scheduler = scheduler
        app.state.watcher = watcher
        app.state.departments = departments
        app.state.weekly = weekly
        app.state.users = users
        app.state.sessions = sessions
        app.state.login_throttle = login_throttle
        app.state.auth_enabled = auth_enabled
        if auth_enabled:
            weak = users.weak_accounts()
            if weak:
                logger.warning(
                    "以下账号仍在使用示例口令，公网部署前请用 scripts/manage_users.py "
                    "改密码：%s",
                    ", ".join(weak),
                )
        # 预热：把 embedding / reranker 的模型加载从"用户的第一问"挪到"启动后台"。
        # 只在显式设置 RAG_WARMUP=1 时开启，避免拖慢测试与 CLI。
        if os.environ.get("RAG_WARMUP", "").strip() == "1":
            def _warmup() -> None:
                started = time.perf_counter()
                try:
                    kb_ = service.kb
                    kb_.embedder.embed_query("预热")
                    if kb_.reranker is not None:
                        kb_.search("预热", top_n=3)
                    logger.info(
                        "预热完成：%.1fs（embedder=%s, reranker=%s）",
                        time.perf_counter() - started,
                        getattr(kb_.embedder, "device", "?"),
                        getattr(kb_.reranker, "device", "?"),
                    )
                except Exception:  # noqa: BLE001 - 预热失败不影响服务
                    logger.warning("预热失败（不影响服务）", exc_info=True)

            threading.Thread(target=_warmup, name="rag-warmup", daemon=True).start()

        logger.info(
            "服务启动：llm=%s（mock=%s）embedder=%s（mock=%s）已入库 chunk=%d 定时同步=%s 目录监听=%s 部门数=%d 周报=%s",
            getattr(service.llm, "model", "unknown"),
            service.llm.is_mock,
            service.kb.embedder.name,
            service.kb.embedder.is_mock,
            service.kb.vector_store.count(),
            scheduler.running,
            watcher.active,
            len(departments.departments),
            weekly.status().get("next_run_at") or ("off" if not weekly.enabled else "pending"),
        )
        try:
            yield
        finally:
            watcher.stop()
            scheduler.shutdown()
            weekly.shutdown()
            try:
                service.close()
            except Exception:  # noqa: BLE001 - 关闭失败不应影响进程退出
                logger.warning("关闭服务时出错", exc_info=True)

    app = FastAPI(
        title="制度与办事流程 RAG 问答助手",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    cors_origins = list(cfg.server.cors_origins or ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        # "*" + allow_credentials 是规范禁止的组合（浏览器会直接拒绝），
        # 通配来源时自动关掉凭证，改了配置也不会踩这个坑
        allow_credentials="*" not in cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 公网暴露时的最低保护（令牌 / 限流）；默认 token 为空、限流 60/min
    # 开了登录就不再叠加共享令牌：否则用户要连续输两次口令
    app.add_middleware(
        AccessControlMiddleware,
        token="" if auth_enabled else cfg.server.public_access_token,
        rate_limit_per_min=cfg.server.rate_limit_per_min,
        dept_token="" if auth_enabled else cfg.dept_console.token,
        trust_forwarded_for=cfg.server.trust_forwarded_for,
    )
    # 后加的中间件在最外层：先鉴权（谁能进），再令牌/限流
    app.add_middleware(
        AuthMiddleware,
        users=users,
        sessions=sessions,
        enabled=auth_enabled,
        cookie_name=cfg.auth.cookie_name,
    )

    app.include_router(auth_router.router)
    app.include_router(health_router.router)
    app.include_router(chat_router.router)
    app.include_router(kb_router.router)
    app.include_router(brand_router.router)
    if cfg.dept_console.enabled:
        app.include_router(dept_router.router)
        app.include_router(console_router.router)

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False, response_model=None)
    def index():
        """面向公众的问答页（手机优先）。"""
        page = WEB_DIR / "index.html"
        if not page.exists():  # pragma: no cover - 前端被删时给可读提示
            return JSONResponse({"detail": "前端页面缺失（src/rag_assistant/web/index.html）"}, status_code=404)
        # no-cache：改完前端刷新就能看到，不会被浏览器缓存成旧页面
        return FileResponse(page, headers={"Cache-Control": "no-store, must-revalidate"})

    @app.get("/dept", include_in_schema=False, response_model=None)
    def dept_console():
        """部门看板：近 N 天各被问了什么、周报预览与投递。"""
        page = WEB_DIR / "dept.html"
        if not page.exists():  # pragma: no cover
            return JSONResponse({"detail": "部门看板页面缺失（dept.html）"}, status_code=404)
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/console", include_in_schema=False, response_model=None)
    def console():
        """运维控制台：索引状态、同步按钮、模式徽标。"""
        page = WEB_DIR / "console.html"
        if not page.exists():  # pragma: no cover
            return JSONResponse({"detail": "控制台页面缺失（console.html）"}, status_code=404)
        return FileResponse(page, headers={"Cache-Control": "no-cache"})

    # ---- 异常映射 ----
    @app.exception_handler(IndexBusyError)
    async def _busy(_: Request, exc: IndexBusyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM 错误：%s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.exception_handler(RagAssistantError)
    async def _app_error(_: Request, exc: RagAssistantError) -> JSONResponse:
        logger.error("业务错误：%s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app


app = create_app()


def run() -> None:  # pragma: no cover - 需要真实端口
    """命令行启动（``rag-api`` / ``python -m rag_assistant.api.main``）。"""
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "rag_assistant.api.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        workers=1,  # 见模块 docstring：多 worker 会各持一份索引
        log_level="info",
        # 放在 Nginx/Caddy 后面时，真实客户端 IP 只在 X-Forwarded-For 里；
        # 不打开这两个开关，限流会把所有访问算成代理那一个 IP。
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":  # pragma: no cover
    run()
