"""知识库同步端到端测试（mock 链路：哈希向量 + 内存向量库 + BM25）。

覆盖需求里"自动更新"的核心承诺：
- 新增文件能检索到；
- 文件内容变了会重建该文档的向量（旧 chunk 被删，不重复召回）；
- 文件删了向量跟着删；
- 重跑同步是幂等的；
- 单个坏文件不影响其他文件。
"""

from __future__ import annotations

from pathlib import Path

from rag_assistant.config import Settings
from rag_assistant.kb import KnowledgeBase


def _expected_docs(settings: Settings) -> int:
    """按同步用的同一套规则扫描 raw 目录，得到"这次应该入库几份文档"。"""
    from rag_assistant.ingestion.manifest import scan_directory

    root = settings.resolve(settings.sync.watch_dir)
    return len(
        scan_directory(
            root,
            include_patterns=settings.sync.include_patterns,
            exclude_dirs=settings.sync.exclude_dirs,
        )
    )


def test_full_sync_and_search(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    report = kb.sync()

    assert report.ok, report.failed
    expected = _expected_docs(mock_settings)
    assert report.added == expected
    assert report.chunks_written > 0
    status = kb.status()
    assert status["documents"] == expected
    assert status["vector_chunks"] == status["bm25_chunks"] == status["manifest_chunks"] > 0

    result = kb.search("在读证明怎么开", top_n=3, threshold=0.0)
    assert result.chunks
    assert result.chunks[0].doc_name == "办事指南-在读证明.txt"
    assert "在读证明" in result.chunks[0].text


def test_sync_is_idempotent(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    second = kb.sync()

    assert second.added == 0
    assert second.modified == 0
    assert second.removed == 0
    assert second.unchanged >= 4
    assert second.chunks_written == 0
    assert kb.status()["vector_chunks"] > 0


def test_sync_rebuilds_when_bm25_index_is_lost(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """BM25 索引损坏/丢空时，sync 要按全量重建，否则关键词召回会一直是空的。

    现实中见过：索引文件被写坏的瞬间刚好被读到，load 失败按空索引处理，
    而文件内容没变，diff 永远是"未变"，于是 BM25 一直空着、检索只剩向量一路。
    """
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    before = kb.bm25.count
    assert before > 0

    kb.bm25.clear()  # 模拟"加载失败 → 空索引"
    report = kb.sync()

    assert report.ok, report.failed
    assert report.rebuild is True
    assert kb.bm25.count == before


def test_modified_file_reindexed(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """改内容 → 只重建该文档，旧 chunk 被删掉（不会新旧并存）。"""
    kb = KnowledgeBase(mock_settings)
    kb.sync()

    target = sample_raw_dir / "学生学籍管理规定.md"
    original = target.read_text(encoding="utf-8")
    target.write_text(original + "\n\n## 第六章 新增条款\n\n**第十五条 新增测试条款**\n\n这是一条新增的制度内容。\n", encoding="utf-8")

    report = kb.sync()
    assert report.modified == 1
    assert report.chunks_deleted > 0
    assert report.chunks_written > 0

    hits = kb.search("新增测试条款", top_n=5, threshold=0.0)
    texts = [chunk.text for chunk in hits.chunks]
    assert any("新增的制度内容" in text for text in texts)
    # 旧版本内容不应残留两份
    assert sum(1 for chunk in hits.chunks if chunk.doc_name == "学生学籍管理规定.md") <= 5


def test_removed_file_cleaned(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    before = kb.status()["vector_chunks"]
    before_docs = kb.status()["documents"]

    (sample_raw_dir / "办事指南-在读证明.txt").unlink()
    report = kb.sync()

    assert report.removed == 1
    assert report.chunks_deleted == 7
    after = kb.status()
    assert after["vector_chunks"] == before - 7
    assert after["documents"] == before_docs - 1
    assert all(chunk.doc_name != "办事指南-在读证明.txt" for chunk in kb.search("在读证明", top_n=10, threshold=0.0).chunks)


def test_rebuild(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    report = kb.sync(rebuild=True)

    assert report.rebuild is True
    assert report.added >= 4
    assert kb.status()["vector_chunks"] > 0


def test_bm25_persisted_and_reloaded(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """BM25 落盘后新实例能直接加载（不必每次重启都重新分词建索引）。"""
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    path = kb.bm25_path
    assert path.exists()
    kb.close()

    reopened = KnowledgeBase(mock_settings)
    assert reopened.bm25.count > 0
    assert reopened.search("差旅费报销", top_n=3, threshold=0.0).chunks


def test_broken_file_isolated(mock_settings: Settings, sample_raw_dir: Path) -> None:
    """坏文件只影响自己：其余照常入库，清单不记录坏文件。"""
    (sample_raw_dir / "损坏.docx").write_bytes(b"not a real docx")

    kb = KnowledgeBase(mock_settings)
    report = kb.sync()

    assert len(report.failed) == 1
    assert "损坏.docx" in report.failed[0]["file"]
    status = kb.status()
    assert status["documents"] == _expected_docs(mock_settings) - 1  # 坏文件之外都入库
    assert status["vector_chunks"] > 0


def test_reset_clears_everything(mock_settings: Settings, sample_raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    kb.sync()
    kb.reset()

    status = kb.status()
    assert status["documents"] == 0
    assert status["vector_chunks"] == 0
    assert status["bm25_chunks"] == 0
    assert kb.search("在读证明", threshold=0.0).chunks == []


def test_empty_directory(mock_settings: Settings, raw_dir: Path) -> None:
    kb = KnowledgeBase(mock_settings)
    report = kb.sync()
    assert report.added == 0
    assert kb.status()["documents"] == 0
