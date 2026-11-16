"""BM25 索引测试：中文分词、检索排序、增删、持久化。"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.retrieval.bm25 import BM25Index, BM25Record, ChineseTokenizer


@pytest.fixture()
def tokenizer(repo_root: Path) -> ChineseTokenizer:
    return ChineseTokenizer(
        userdict_path=repo_root / "config" / "jieba_userdict.txt",
        stopwords_path=repo_root / "config" / "stopwords_zh.txt",
    )


@pytest.fixture()
def records() -> list[BM25Record]:
    return [
        BM25Record("d1#1", "学生休学应当办理休学手续，休学期一般以一年为期，累计不得超过两年。", {"doc_id": "d1"}),
        BM25Record("d2#1", "在读证明可线上办理，1 个工作日内审核，首次开具免费。", {"doc_id": "d2"}),
        BM25Record("d3#1", "科研经费差旅费报销须在 15 个工作日内办理，住宿费按标准限额报销。", {"doc_id": "d3"}),
    ]


# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------
def test_tokenizer_keeps_domain_terms(tokenizer: ChineseTokenizer) -> None:
    """学院术语不被切碎（自定义词典生效）。"""
    tokens = tokenizer.tokenize("教务处负责学籍管理和在读证明开具")
    assert "教务处" in tokens
    assert "学籍管理" in tokens  # 词典里有"学籍管理"，不会被切成"学籍/管理"
    assert "在读证明" in tokens


def test_tokenizer_filters_stopwords_and_single_chars(tokenizer: ChineseTokenizer) -> None:
    """停用词与中文单字被过滤，数字保留。"""
    tokens = tokenizer.tokenize("学生的休学流程是 15 个工作日吗")
    assert "的" not in tokens
    assert "是" not in tokens
    assert "15" in tokens
    assert all(len(token) >= 2 or token.isalnum() for token in tokens)


def test_tokenizer_keeps_order_and_duplicates(tokenizer: ChineseTokenizer) -> None:
    """保留重复词（BM25 需要词频）。"""
    tokens = tokenizer.tokenize("休学 休学 复学")
    assert tokens.count("休学") == 2


def test_tokenizer_empty(tokenizer: ChineseTokenizer) -> None:
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("，。！") == []


# --------------------------------------------------------------------------
# 检索
# --------------------------------------------------------------------------
def test_search_ranks_matching_doc_first(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    """关键词命中者排第一，且分数为正。"""
    index = BM25Index(tokenizer)
    index.add(records)
    hits = index.search("休学多久", top_k=3)
    assert hits
    assert hits[0].chunk_id == "d1#1"
    assert hits[0].score > 0


def test_search_respects_top_k(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    index = BM25Index(tokenizer)
    index.add(records)
    assert len(index.search("办理", top_k=1)) == 1


def test_search_no_match_returns_empty(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    """完全不相关的查询不应返回零分噪声（否则 RRF 会给它虚假名次）。"""
    index = BM25Index(tokenizer)
    index.add(records)
    assert index.search("图书馆开放时间", top_k=5) == []


def test_search_empty_query(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    index = BM25Index(tokenizer)
    index.add(records)
    assert index.search("", top_k=5) == []
    assert index.search("，。", top_k=5) == []


def test_add_overwrites_same_chunk(tokenizer: ChineseTokenizer) -> None:
    """同 chunk_id 覆盖，不产生重复条目。"""
    index = BM25Index(tokenizer)
    index.add([BM25Record("d1#1", "旧的休学内容", {"doc_id": "d1"})])
    index.add([BM25Record("d1#1", "新的复学内容", {"doc_id": "d1"})])
    assert index.count == 1
    assert index.search("复学", top_k=1)[0].text == "新的复学内容"


def test_remove_docs(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    """按 doc_id 删除后不再被召回。"""
    index = BM25Index(tokenizer)
    index.add(records)
    assert index.remove_docs(["d1"]) == 1
    assert index.count == 2
    assert all(hit.metadata["doc_id"] != "d1" for hit in index.search("休学", top_k=3))
    assert index.doc_ids() == {"d2", "d3"}
    assert index.remove_docs([]) == 0


def test_clear(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    index = BM25Index(tokenizer)
    index.add(records)
    index.clear()
    assert index.count == 0
    assert index.search("休学", top_k=3) == []


def test_incremental_rebuild(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    """增删后检索结果跟着变（惰性重建生效）。"""
    index = BM25Index(tokenizer)
    index.add(records)
    assert index.search("休学", top_k=1)[0].chunk_id == "d1#1"

    index.remove_docs(["d1"])
    index.add([BM25Record("d4#1", "休学申请须经二级学院审核后报教务处审批。", {"doc_id": "d4"})])
    hits = index.search("休学", top_k=2)
    assert hits[0].chunk_id == "d4#1"


# --------------------------------------------------------------------------
# 持久化
# --------------------------------------------------------------------------
def test_save_and_load(tokenizer: ChineseTokenizer, records: list[BM25Record], tmp_path: Path) -> None:
    """落盘再读回，检索结果与分数一致。"""
    path = tmp_path / "bm25.json"
    index = BM25Index(tokenizer, storage_path=path)
    index.add(records)
    index.save()

    before = index.search("差旅费报销", top_k=2)

    restored = BM25Index(tokenizer, storage_path=path)
    assert restored.load() == 3
    after = restored.search("差旅费报销", top_k=2)

    assert [hit.chunk_id for hit in before] == [hit.chunk_id for hit in after]
    assert before[0].score == pytest.approx(after[0].score, rel=1e-9)


def test_load_missing_file_is_empty(tokenizer: ChineseTokenizer, tmp_path: Path) -> None:
    index = BM25Index(tokenizer, storage_path=tmp_path / "nope.json")
    assert index.load() == 0
    assert index.count == 0


def test_load_corrupt_file(tokenizer: ChineseTokenizer, tmp_path: Path) -> None:
    """损坏文件按空索引处理（下次同步会重建），不崩。"""
    path = tmp_path / "bm25.json"
    path.write_text("{broken", encoding="utf-8")
    index = BM25Index(tokenizer, storage_path=path)
    assert index.load() == 0


def test_statistics(tokenizer: ChineseTokenizer, records: list[BM25Record]) -> None:
    index = BM25Index(tokenizer)
    index.add(records)
    stats = index.statistics()
    assert stats["chunks"] == 3
    assert stats["unique_tokens"] > 10
    assert stats["avg_tokens"] > 0
