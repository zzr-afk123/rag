"""部门目录 / 统计 / 周报 / 看板接口 的测试。

覆盖点：
- 关键词命中（含同义词、多部门、无命中、上限截断、命中但没联系方式被过滤）；
- 配置容错（文件缺失、YAML 坏掉都退化为空目录，服务照常启动）；
- 聚合口径（按字面去重、部门归属优先用日志字段、无时间戳时按文件名/mtime 兜底）；
- 周报 Markdown 与投递（默认 dry_run 只写 .eml，不碰网络）；
- 看板接口（/dept/summary、/dept/report、/dept/report/send、令牌校验）。
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from rag_assistant.api.main import create_app
from rag_assistant.chat_service import ChatService
from rag_assistant.config import Settings
from rag_assistant.departments import DepartmentDirectory
from rag_assistant.insights import (
    deliver_weekly,
    load_records,
    summarize,
    weekly_markdown,
    weekly_workbook,
)
from rag_assistant.kb import KnowledgeBase

REPO_DEPARTMENTS = Path(__file__).resolve().parents[1] / "config" / "departments.yaml"


# --------------------------------------------------------------------------
# 部门目录
# --------------------------------------------------------------------------
def test_load_real_directory() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    assert len(directory.departments) >= 5
    jwc = directory.by_id("jwc")
    assert jwc is not None and jwc.name == "教务处" and jwc.phone
    assert jwc.has_contact


def test_match_by_institution_keyword() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    hits = directory.match("休学最长可以多久？")
    assert [item.id for item in hits][:1] == ["jwc"]


def test_match_by_domain_synonyms() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    assert directory.match("差旅费报销要附发票吗")[0].id == "cwc"
    assert directory.match("图书馆自习座位怎么约")[0].id == "lib"
    assert directory.match("剧毒品应该怎么存放")[0].id == "lab"


def test_match_returns_empty_for_offtopic() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    assert directory.match("郑州今天天气怎么样？") == []
    assert directory.match("") == []


def test_match_caps_suggestions() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    directory.max_suggestions = 2  # type: ignore[misc]
    hits = directory.match("休学的学籍规定和报销发票、图书馆借书")
    assert len(hits) <= 2


def test_rewritten_query_counts_less_than_question() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    hits = directory.match("这个怎么办？", rewritten="休学需要什么材料")
    assert [item.id for item in hits] == ["jwc"]


def test_doc_name_contributes_but_does_not_override_question() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    hits = directory.match("休学最长多久", doc_names=["科研经费报销管理办法.docx"])
    assert hits and hits[0].id == "jwc", "问题命中的部门应排第一"


def test_missing_file_degrades_quietly(tmp_path: Path) -> None:
    directory = DepartmentDirectory.load(tmp_path / "nope.yaml")
    assert directory.enabled is False
    assert directory.match("休学") == []
    assert directory.contacts_for("休学") == []


def test_broken_yaml_degrades_quietly(tmp_path: Path) -> None:
    path = tmp_path / "departments.yaml"
    path.write_text("departments: [oops", encoding="utf-8")
    directory = DepartmentDirectory.load(path)
    assert directory.enabled is False


def test_department_without_contact_is_filtered(tmp_path: Path) -> None:
    path = tmp_path / "departments.yaml"
    path.write_text(
        "departments:\n"
        "  - id: ghost\n    name: 无联系方式部门\n    keywords: [神秘]\n",
        encoding="utf-8",
    )
    directory = DepartmentDirectory.load(path)
    assert directory.match("神秘的问题") == []
    assert [item.id for item in directory.match("神秘的问题", only_with_contact=False)] == ["ghost"]


def test_to_dict_hides_keywords() -> None:
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    payload = directory.by_id("jwc").to_dict()  # type: ignore[union-attr]
    assert payload["phone"] == directory.by_id("jwc").phone  # type: ignore[union-attr]
    assert "keywords" not in payload


# --------------------------------------------------------------------------
# 统计口径
# --------------------------------------------------------------------------
def write_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_summarize_counts_and_departments(tmp_path: Path) -> None:
    now = datetime(2026, 11, 11, 10, 0, 0)
    path = tmp_path / "logs" / "qa.jsonl"
    write_records(
        path,
        [
            {"ts": "2026-11-10T09:00:00", "question": "休学最长可以多久？", "refused": False, "departments": ["jwc"]},
            {"ts": "2026-11-10T09:05:00", "question": "休学最长可以多久", "refused": False, "departments": ["jwc"]},
            {"ts": "2026-11-09T09:00:00", "question": "差旅费报销要附什么发票", "refused": True, "departments": ["cwc"]},
            {"ts": "2026-11-09T10:00:00", "question": "郑州天气怎么样", "refused": True, "departments": []},
            {"ts": "2026-08-01T10:00:00", "question": "太久了不该统计", "refused": False, "departments": ["jwc"]},
        ],
    )
    records = load_records(tmp_path / "logs", days=30, now=now)
    assert len(records) == 4, "窗口外的记录应被过滤"

    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    summary = summarize(records, directory=directory, days=30, now=now)

    assert summary["total"] == 4
    assert summary["refused"] == 2
    assert summary["unique_questions"] == 3  # "休学最长可以多久？"与"…多久"同一条
    assert summary["unmatched"] == 1
    depts = {row["id"]: row for row in summary["departments"]}
    assert depts["jwc"]["count"] == 2
    assert depts["cwc"]["refused"] == 1
    assert summary["top_questions"][0]["count"] == 2
    assert len(summary["daily"]) == 30


def test_records_without_ts_fall_back_to_filename(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    rotated = log_dir / "qa.jsonl.2026-11-09"
    rotated.write_text(json.dumps({"question": "休学", "refused": False}, ensure_ascii=False) + "\n", encoding="utf-8")
    records = load_records(log_dir, days=30, now=datetime(2026, 11, 11))
    assert len(records) == 1
    assert records[0]["_day"] == "2026-11-09"


def test_department_inferred_from_keywords_when_field_missing(tmp_path: Path) -> None:
    now = datetime(2026, 11, 11)
    path = tmp_path / "qa.jsonl"
    write_records(path, [{"ts": "2026-11-10T09:00:00", "question": "图书馆几点开门", "refused": False}])
    records = load_records(tmp_path, days=30, now=now)
    summary = summarize(records, directory=DepartmentDirectory.load(REPO_DEPARTMENTS), days=30, now=now)
    assert [row["id"] for row in summary["departments"]] == ["lib"]


# --------------------------------------------------------------------------
# 周报与投递
# --------------------------------------------------------------------------
def _summary(now: datetime) -> dict:
    path_rows = [
        {"ts": "2026-11-10T09:00:00", "question": "休学最长可以多久？", "refused": False, "departments": ["jwc"]},
        {"ts": "2026-11-10T09:05:00", "question": "休学最长可以多久", "refused": False, "departments": ["jwc"]},
        {"ts": "2026-11-10T09:09:00", "question": "复学要交什么材料", "refused": True, "departments": ["jwc"]},
    ]
    records = [
        {**row, "_ts": row["ts"], "_day": row["ts"][:10]} for row in path_rows
    ]
    return summarize(records, directory=DepartmentDirectory.load(REPO_DEPARTMENTS), days=7, now=now)


def test_weekly_markdown_contains_department_view() -> None:
    now = datetime(2026, 11, 11, 9, 0, 0)
    summary = _summary(now)
    directory = DepartmentDirectory.load(REPO_DEPARTMENTS)
    markdown = weekly_markdown(summary, department=directory.by_id("jwc"), school_name="郑州大学")
    assert "郑州大学" in markdown
    assert "教务处" in markdown
    assert "休学最长可以多久" in markdown
    assert "建议关注" in markdown


def test_weekly_markdown_overall_view() -> None:
    now = datetime(2026, 11, 11, 9, 0, 0)
    markdown = weekly_markdown(_summary(now), school_name="郑州大学")
    assert "部门分布" in markdown
    assert "高频问题" in markdown


def test_deliver_weekly_dry_run_writes_eml(mock_settings: Settings, tmp_path: Path) -> None:
    settings = replace(
        mock_settings,
        report=replace(mock_settings.report, enabled=True, dry_run=True, days=7),
    )
    write_records(
        settings.log_dir / "qa.jsonl",
        [
            {"ts": datetime.now().isoformat(timespec="seconds"), "question": "休学最长可以多久", "refused": False, "departments": ["jwc"]},
            {"ts": datetime.now().isoformat(timespec="seconds"), "question": "差旅费报销要附发票吗", "refused": True, "departments": ["cwc"]},
        ],
    )
    result = deliver_weekly(settings)
    assert result["dry_run"] is True
    statuses = {item["department_id"]: item["status"] for item in result["deliveries"]}
    assert statuses["jwc"] == "dry_run"
    assert statuses["cwc"] == "dry_run"
    assert statuses["lib"] == "skipped_empty"

    files = list(settings.outbox_dir.glob("*.eml"))
    assert files, "dry_run 应把邮件写进 outbox"
    raw = files[0].read_bytes().decode("utf-8", "replace")
    assert "Subject:" in raw
    assert (settings.outbox_dir / files[0].name.replace(".eml", ".md")).exists()


def test_deliver_weekly_without_smtp_is_dry_run(mock_settings: Settings) -> None:
    settings = replace(mock_settings, report=replace(mock_settings.report, enabled=False, smtp_host=""))
    result = deliver_weekly(settings, days=7)
    assert result["dry_run"] is True
    assert result["sent"] == 0


# --------------------------------------------------------------------------
# 看板接口
# --------------------------------------------------------------------------
@pytest.fixture()
def dept_client(mock_settings: Settings):
    """不建索引：部门看板只读日志，不需要向量库有内容。"""
    (mock_settings.log_dir).mkdir(parents=True, exist_ok=True)
    write_records(
        mock_settings.log_dir / "qa.jsonl",
        [
            {"ts": datetime.now().isoformat(timespec="seconds"), "question": "休学最长可以多久", "refused": False, "departments": ["jwc"]},
            {"ts": datetime.now().isoformat(timespec="seconds"), "question": "图书馆周五几点开门", "refused": True, "departments": ["lib"]},
        ],
    )
    kb = KnowledgeBase(mock_settings)
    service = ChatService(mock_settings, kb=kb)
    app = create_app(mock_settings, chat_service=service)
    with TestClient(app) as client:
        yield client, mock_settings


def test_brand_endpoint_is_public(dept_client) -> None:
    client, _settings = dept_client
    payload = client.get("/brand").json()
    assert payload["school_name"] == "郑州大学"
    assert payload["motto"]
    assert payload["departments"], "应返回部门联系方式"
    assert "email" not in payload["departments"][0], "公开接口不暴露邮箱"


def test_dept_page_and_summary(dept_client) -> None:
    client, _settings = dept_client
    page = client.get("/dept")
    assert page.status_code == 200
    assert "部门看板" in page.text
    assert "/static/emblem-white.png" in page.text

    payload = client.get("/dept/summary?days=30").json()
    assert payload["total"] == 2
    assert payload["window_days"] == 30
    assert payload["brand"]["school_name"] == "郑州大学"
    assert {row["id"] for row in payload["departments"]} == {"jwc", "lib"}
    assert payload["report"]["enabled"] is False
    assert payload["daily"]


def test_dept_report_markdown(dept_client) -> None:
    client, _settings = dept_client
    response = client.get("/dept/report?dept=jwc&days=7")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    assert "教务处" in response.text
    assert client.get("/dept/report?dept=nope").status_code == 404


def test_dept_report_download_is_excel(dept_client) -> None:
    """下载给的是 xlsx：中文校名不能直接进 Content-Disposition，回退名必须是 ASCII。"""
    client, _settings = dept_client
    response = client.get("/dept/report?dept=jwc&days=7&download=true")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert "filename*=UTF-8''" in disposition
    # filename= 那个回退名必须是纯 ASCII，且扩展名是 .xlsx
    fallback = disposition.split("filename=")[1].split(";")[0].strip('"')
    fallback.encode("ascii")  # 不抛异常即合格
    assert fallback.endswith(".xlsx")

    wb = load_workbook(BytesIO(response.content))
    assert wb.sheetnames[0] == "概况"
    assert "高频问题" in wb.sheetnames
    assert "建议关注" in wb.sheetnames
    # 部门视角的标题里应出现部门名
    assert "教务处" in str(wb["概况"]["A1"].value)


def test_weekly_workbook_school_sheets(dept_client) -> None:
    """全校周报要有部门分布、高频问题、未答上、每日趋势四张表，且合计行存在。"""
    client, settings = dept_client
    directory = DepartmentDirectory.from_settings(settings)
    records = load_records(settings.log_dir, qa_log=settings.logging.qa_log, days=7)
    summary = summarize(records, directory=directory, days=7)
    data = weekly_workbook(
        summary,
        school_name=settings.branding.school_name,
        dept_names={row.id: row.name for row in directory.departments},
    )
    wb = load_workbook(BytesIO(data))
    assert wb.sheetnames == ["概况", "部门分布", "高频问题", "未答上的问题", "每日趋势"]
    dept_ws = wb["部门分布"]
    assert dept_ws["A4"].value == "部门"
    # 最后一行是合计（有部门行时）
    last = dept_ws.max_row
    if dept_ws.cell(row=last, column=1).value == "合计":
        assert dept_ws.cell(row=last, column=2).value == summary["total"]
    overview = wb["概况"]
    labels = [overview.cell(row=row, column=1).value for row in range(6, 12)]
    assert "咨询总量" in labels
    assert "拒答率" in labels


def test_dept_send_dry_run(dept_client) -> None:
    client, _settings = dept_client
    result = client.post("/dept/report/send?dept=jwc&days=7").json()
    assert result["dry_run"] is True
    assert result["deliveries"][0]["department_id"] == "jwc"


def test_dept_department_detail(dept_client) -> None:
    client, _settings = dept_client
    payload = client.get("/dept/department/jwc?days=30").json()
    assert payload["department"]["name"] == "教务处"
    assert payload["summary"]["count"] >= 1
    assert client.get("/dept/department/nope").json()["error"]


def test_dept_token_is_enforced(mock_settings: Settings) -> None:
    settings = replace(
        mock_settings,
        dept_console=replace(mock_settings.dept_console, token="dept-secret"),
    )
    (settings.log_dir).mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(settings)
    service = ChatService(settings, kb=kb)
    app = create_app(settings, chat_service=service)
    with TestClient(app) as client:
        assert client.get("/dept/summary").status_code == 401
        assert client.get("/dept/summary?token=dept-secret").status_code == 200
        assert client.get("/dept/summary", headers={"X-Access-Token": "dept-secret"}).status_code == 200
        # 页面本身要能打开，否则用户没法填令牌
        assert client.get("/dept").status_code == 200


# --------------------------------------------------------------------------
# 静态资源与品牌（手机访问 / 主屏图标）
# --------------------------------------------------------------------------
def test_static_brand_assets(dept_client) -> None:
    client, _settings = dept_client
    logo = client.get("/static/logo.png")
    assert logo.status_code == 200
    assert "image/png" in logo.headers["content-type"]
    emblem = client.get("/static/emblem.png")
    assert emblem.status_code == 200
    assert "image/png" in emblem.headers["content-type"]

    manifest = client.get("/static/manifest.webmanifest")
    assert manifest.status_code == 200
    assert "制度问答" in manifest.text


def test_index_page_has_brand_and_contact_slot(dept_client) -> None:
    client, _settings = dept_client
    page = client.get("/").text
    assert "/static/logo.png" in page
    assert "manifest.webmanifest" in page
    assert "相关部门联系方式" in page
    assert "apple-mobile-web-app-capable" in page


def test_health_exposes_public_url(dept_client) -> None:
    client, _settings = dept_client
    payload = client.get("/health").json()
    assert payload["public_url"] == ""  # 未配置时为空串，字段必须存在
