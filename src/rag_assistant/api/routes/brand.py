"""公开品牌信息：校名、校训、配色、部门目录的"联系方式"部分。

为什么单独一个接口：前端页面要用配置里的校名/主色渲染，而不是把"郑州大学"
写死进 HTML——换学校只改 `settings.yaml`，不动前端。这个接口只返回公开信息，
不带任何统计数据（统计在 `/dept/*`，需要令牌）。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...config import Settings
from ...departments import DepartmentDirectory

router = APIRouter(tags=["brand"])


@router.get("/brand", summary="品牌与公开的部门联系方式")
def brand(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    directory: DepartmentDirectory = request.app.state.departments
    branding = settings.branding
    return {
        "school_name": branding.school_name,
        "short_name": branding.short_name,
        "motto": branding.motto,
        "site_title": branding.site_title,
        "primary_color": branding.primary_color,
        "accent_color": branding.accent_color,
        "footer_note": branding.footer_note,
        # 公开页面只给电话/地点/时间；邮箱留给 /dept（部门内部看板）用，减少被批量爬取
        "departments": [
            {
                "id": item["id"],
                "name": item["name"],
                "phone": item.get("phone", ""),
                "address": item.get("address", ""),
                "hours": item.get("hours", ""),
            }
            for item in directory.catalog()
        ],
    }
