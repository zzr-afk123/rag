"""同步 CLI 的集成测试（含"清单 / 向量库 / BM25 三者一致"校验）。

这一层测的是"用户实际敲的那条命令"：``python scripts/sync.py``。
同步逻辑本身的单元行为在 ``test_kb_sync.py`` 里逐项覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sync as sync_cli
from rag_assistant.config import Settings
from rag_assistant.ingestion.manifest import load_manifest
from rag_assistant.kb import KnowledgeBase


def _assert_three_way_consistent(settings: Settings) -> None:
    """清单、向量库、BM25 的数量必须完全一致（否则会出现"能搜到但引用不到"）。"""
    kb = KnowledgeBase(settings)
    manifest = load_manifest(settings.manifest_path)
    try:
        assert manifest.chunk_count == kb.vector_store.count()
        assert kb.bm25.count == kb.vector_store.count()
        assert manifest.doc_count == len(kb.vector_store.list_doc_ids())
    finally:
        kb.close()


def test_cli_full_sync(mock_settings: Settings, sample_raw_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = sync_cli.main(["--rebuild"], settings=mock_settings)
    out = capsys.readouterr().out

    assert code == 0
    assert "同步结果" in out
    assert "写入 chunk" in out
    assert "mock" in out  # 明确提示当前是 mock 向量
    _assert_three_way_consistent(mock_settings)


def test_cli_second_run_is_noop(mock_settings: Settings, sample_raw_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync_cli.main([], settings=mock_settings)
    capsys.readouterr()
    code = sync_cli.main([], settings=mock_settings)
    out = capsys.readouterr().out

    assert code == 0
    assert "新增 0" in out
    assert "索引状态" in out


def test_cli_status(mock_settings: Settings, sample_raw_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sync_cli.main([], settings=mock_settings)
    capsys.readouterr()

    assert sync_cli.main(["--status"], settings=mock_settings) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] >= 4
    assert payload["vector_chunks"] > 0
    assert payload["collection"].startswith("kb__")


def test_cli_json_output(mock_settings: Settings, sample_raw_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert sync_cli.main(["--json"], settings=mock_settings) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"]["added"] >= 4
    assert payload["status"]["documents"] >= 4


def test_cli_reports_failure_exit_code(
    mock_settings: Settings, sample_raw_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """有文件解析失败时返回非 0（方便 cron / CI 报警）。"""
    (sample_raw_dir / "坏文件.docx").write_bytes(b"definitely not a docx")
    code = sync_cli.main(["--rebuild"], settings=mock_settings)
    out = capsys.readouterr().out

    assert code == 1
    assert "[失败]" in out
    assert "坏文件.docx" in out


def test_cli_limit(mock_settings: Settings, sample_raw_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--limit 只处理部分文件（调试大目录时用）。"""
    assert sync_cli.main(["--limit", "2"], settings=mock_settings) == 0
    capsys.readouterr()

    kb = KnowledgeBase(mock_settings)
    try:
        assert kb.status()["documents"] == 2
    finally:
        kb.close()
