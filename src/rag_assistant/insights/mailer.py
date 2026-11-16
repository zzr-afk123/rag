"""周报投递：把「本部门这周被问了什么」发给部门联系人。

为什么默认 `dry_run=true`：自动发邮件是**外部副作用**，配置错一次就会给真实
部门发空报告。所以默认只把邮件写进 `logs/outbox/*.eml`（可用 Foxmail/Outlook
直接打开预览），确认内容无误后再把 `report.dry_run` 关掉、配好 SMTP。

投递失败绝不抛到调度线程：每个部门独立 try，结果逐条记录在返回值里，
看板与日志能看到"谁没发出去、为什么"。
"""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, formatdate
from pathlib import Path
from typing import Any

from ..departments import Department, DepartmentDirectory
from ..logging_setup import get_logger
from .analytics import dept_slice, load_records, summarize, weekly_markdown

logger = get_logger("insights.mailer")


@dataclass
class Delivery:
    """一个部门的投递结果。"""

    department_id: str
    name: str
    email: str
    count: int
    status: str  # sent | dry_run | skipped_no_email | skipped_empty | failed
    file: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "department_id": self.department_id,
            "name": self.name,
            "email": self.email,
            "count": self.count,
            "status": self.status,
        }
        if self.file:
            payload["file"] = self.file
        if self.error:
            payload["error"] = self.error
        return payload


def build_email(
    *,
    subject: str,
    to_addr: str,
    body: str,
    from_addr: str,
    school_name: str = "",
) -> EmailMessage:
    """构造纯文本邮件（制度类内容不需要 HTML，纯文本更不会被当成营销邮件）。"""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((school_name or "制度问答系统", from_addr)) if from_addr else (school_name or "noreply@localhost")
    message["To"] = to_addr
    message["Date"] = formatdate(localtime=True)
    message.set_content(body)
    return message


def _send(message: EmailMessage, settings) -> None:
    config = settings.report
    password = ""
    if config.smtp_password_env:
        import os

        password = os.environ.get(config.smtp_password_env, "")
    if config.smtp_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(config.smtp_host, int(config.smtp_port), context=context, timeout=20) as server:
            if config.smtp_user:
                server.login(config.smtp_user, password)
            server.send_message(message)
        return
    with smtplib.SMTP(config.smtp_host, int(config.smtp_port), timeout=20) as server:
        server.ehlo()
        try:
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        except smtplib.SMTPNotSupportedError:  # pragma: no cover - 内网明文 SMTP
            logger.warning("SMTP 服务不支持 STARTTLS，按明文发送")
        if config.smtp_user:
            server.login(config.smtp_user, password)
        server.send_message(message)


def _write_outbox(
    settings,
    department: Department,
    subject: str,
    markdown: str,
    message: EmailMessage,
    *,
    moment: datetime,
) -> Path:
    outbox = settings.outbox_dir
    outbox.mkdir(parents=True, exist_ok=True)
    stamp = moment.strftime("%Y%m%d-%H%M")
    base = f"{stamp}-{department.id}"
    (outbox / f"{base}.md").write_text(markdown, encoding="utf-8")
    target = outbox / f"{base}.eml"
    target.write_bytes(bytes(message))
    return target


def deliver_weekly(
    settings,
    *,
    directory: DepartmentDirectory | None = None,
    days: int | None = None,
    now: datetime | None = None,
    dry_run: bool | None = None,
    only: str | None = None,
) -> dict[str, Any]:
    """生成并投递周报。

    Args:
        settings: 全局配置（用到 log_dir / report / branding）。
        directory: 部门目录；缺省按配置加载。
        days: 统计窗口，默认 `report.days`（7 天）。
        dry_run: 覆盖 `report.dry_run`；None 表示按配置。
        only: 只发某个部门（调试/手动补发）。
    """
    moment = now or datetime.now()
    config = settings.report
    window = int(days if days is not None else config.days)
    department_dir = directory or DepartmentDirectory.from_settings(settings)
    simulate = config.dry_run if dry_run is None else bool(dry_run)

    records = load_records(
        settings.log_dir,
        qa_log=settings.logging.qa_log,
        days=window,
        now=moment,
    )
    summary = summarize(
        records,
        directory=department_dir,
        days=window,
        now=moment,
        top_n=settings.dept_console.top_n,
    )

    deliveries: list[Delivery] = []
    for department in department_dir.departments:
        if only and department.id != only:
            continue
        row = dept_slice(summary, department.id)
        count = int(row.get("count", 0))
        if not department.email:
            deliveries.append(
                Delivery(department.id, department.name, "", count, "skipped_no_email")
            )
            continue
        if count == 0:
            deliveries.append(
                Delivery(department.id, department.name, department.email, 0, "skipped_empty")
            )
            continue

        markdown = weekly_markdown(
            summary,
            department=department,
            school_name=settings.branding.school_name,
            site_title=settings.branding.site_title,
            example_data=True,
        )
        subject = f"{config.subject_prefix}{settings.branding.school_name}{department.name}近 {window} 天咨询"
        message = build_email(
            subject=subject,
            to_addr=department.email,
            body=markdown,
            from_addr=config.from_addr or config.smtp_user,
            school_name=settings.branding.school_name,
        )

        if simulate or not config.smtp_host:
            try:
                target = _write_outbox(settings, department, subject, markdown, message, moment=moment)
                deliveries.append(
                    Delivery(department.id, department.name, department.email, count, "dry_run", str(target))
                )
            except OSError as exc:
                logger.warning("周报写入 outbox 失败：%s", exc)
                deliveries.append(
                    Delivery(department.id, department.name, department.email, count, "failed", error=str(exc))
                )
            continue

        try:
            _send(message, settings)
            deliveries.append(Delivery(department.id, department.name, department.email, count, "sent"))
            logger.info("周报已发送：%s → %s（%d 条咨询）", department.name, department.email, count)
        except Exception as exc:  # noqa: BLE001 - 单个部门失败不能影响其它部门
            logger.exception("周报发送失败：%s → %s", department.name, department.email)
            deliveries.append(
                Delivery(department.id, department.name, department.email, count, "failed", error=str(exc))
            )

    sent = sum(1 for item in deliveries if item.status == "sent")
    logger.info(
        "周报处理完成：窗口 %d 天，记录 %d 条，发送 %d 封，模拟/落盘 %d 封",
        window,
        len(records),
        sent,
        sum(1 for item in deliveries if item.status == "dry_run"),
    )
    return {
        "generated_at": moment.replace(microsecond=0).isoformat(timespec="seconds"),
        "window_days": window,
        "dry_run": simulate or not config.smtp_host,
        "records": len(records),
        "total": summary.get("total", 0),
        "sent": sent,
        "deliveries": [item.to_dict() for item in deliveries],
    }


__all__ = ["Delivery", "build_email", "deliver_weekly"]
