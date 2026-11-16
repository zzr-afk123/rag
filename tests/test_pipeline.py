"""摄入流水线测试：试算不写盘、异常按文件隔离、报告可序列化。"""

from __future__ import annotations

import json
from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.ingestion.pipeline import IngestPipeline


def test_dry_run_all_formats(settings: Settings, sample_raw_dir: Path) -> None:
    """四种格式都能解析切块，且不产生任何索引文件。"""
    pipeline = IngestPipeline(settings)
    report = pipeline.dry_run()

    assert not report.failed, report.to_dict()["failed"]
    assert report.stats.docs >= 4
    assert report.stats.chunks > 10
    assert report.stats.table_chunks >= 3  # MD 一张 + XLSX 两张
    assert report.stats.tokens > 0

    # 试算不得写盘
    assert not settings.manifest_path.exists()
    assert not settings.chroma_dir.exists()


def test_report_serializable(settings: Settings, sample_raw_dir: Path) -> None:
    """报告能直接 JSON 化（供 API / CI 上报）。"""
    report = IngestPipeline(settings).dry_run()
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "files" in payload

    text = report.render()
    assert "试算结果" in text
    assert "合计" in text


def test_limit(settings: Settings, sample_raw_dir: Path) -> None:
    """--limit 只处理前 N 个文件。"""
    report = IngestPipeline(settings).dry_run(limit=2)
    assert len(report.results) == 2


def test_broken_file_isolated(settings: Settings, sample_raw_dir: Path) -> None:
    """一个坏文件不应中断整批处理。"""
    (sample_raw_dir / "损坏文件.docx").write_bytes(b"this is not a docx")
    report = IngestPipeline(settings).dry_run()

    assert len(report.failed) == 1
    assert "损坏文件.docx" in report.failed[0].rel_path
    assert len(report.ok_results) >= 4  # 其余文件照常处理


def test_empty_raw_dir(settings: Settings, raw_dir: Path) -> None:
    """空目录：报告为空但不报错。"""
    report = IngestPipeline(settings).dry_run()
    assert report.stats.docs == 0
    assert not report.failed
