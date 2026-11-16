"""路由公共依赖：从 ``app.state`` 取服务实例。

为什么用 ``app.state`` 而不是模块级全局：测试可以给每个 app 注入不同的
``ChatService``（临时目录、mock 链路），互不干扰。
"""

from __future__ import annotations

from fastapi import Request

from ..chat_service import ChatService
from ..config import Settings
from ..kb import KnowledgeBase
from .jobs import SyncJobManager


def get_chat(request: Request) -> ChatService:
    return request.app.state.chat


def get_kb(request: Request) -> KnowledgeBase:
    return request.app.state.chat.kb


def get_jobs(request: Request) -> SyncJobManager:
    return request.app.state.jobs


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings
