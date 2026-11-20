"""同义词归一与多义词过滤的单元测试。

对应一次真实事故：用户问"我是想要进校访问"，系统答了《图书馆管理办法》里的
"校外访问电子数据库需校园 VPN 认证"。问的是**人进校**，答的是**访问系统**。

两条修复各自单独可测：
1. 同义词归一：把「进校」换成制度里用的「入校」，BM25 与词面覆盖才可能命中；
2. 多义词过滤：「访问 / 通行 / 办理」这类词在每份制度里都出现，
   不能当作"问的就是这个主题"的证据，否则覆盖率永远虚高。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.retrieval.query import SynonymTable
from rag_assistant.retrieval.retriever import HybridRetriever


class _StubTokenizer:
    """按空格切词的假分词器，测试不依赖 jieba 的实际切分结果。"""

    def tokenize(self, text: str) -> list[str]:
        return [part for part in text.split() if part]


class _StubBm25:
    def __init__(self) -> None:
        self.tokenizer = _StubTokenizer()


def test_load_parses_groups_menu(tmp_path: Path) -> None:
    path = tmp_path / "synonyms.yaml"
    path.write_text(
        "groups:\n"
        "  - canonical: 入校\n"
        "    variants: [进校, 进校园]\n"
        "  - canonical: 访客\n"
        "    variants: [来访]\n",
        encoding="utf-8",
    )
    table = SynonymTable.load(path)
    assert table.normalize("我想进校看看") == "我想入校看看"
    assert table.normalize("有人来访") == "有人访客"
    assert table.matched("我想进校看看") == [("进校", "入校")]


def test_longest_variant_wins(tmp_path: Path) -> None:
    path = tmp_path / "synonyms.yaml"
    path.write_text(
        "groups:\n"
        "  - canonical: 入校\n"
        "    variants: [进校, 进校园]\n",
        encoding="utf-8",
    )
    table = SynonymTable.load(path)
    # 若「进校」先于「进校园」替换，"进校园"会被切成"入校园"这种怪词
    assert table.normalize("进校园参观") == "入校参观"


def test_missing_file_is_identity(tmp_path: Path) -> None:
    table = SynonymTable.load(tmp_path / "nope.yaml")
    assert table.normalize("进校访问") == "进校访问"
    assert table.matched("进校访问") == []


def test_broken_file_is_identity(tmp_path: Path) -> None:
    path = tmp_path / "synonyms.yaml"
    path.write_text("groups: [this is not: valid", encoding="utf-8")
    table = SynonymTable.load(path)
    assert table.normalize("进校访问") == "进校访问"


def test_repo_synonyms_covers_incident_words() -> None:
    """仓库自带的词表里必须有这次事故涉及的两组。"""
    project = Path(__file__).resolve().parents[1]
    table = SynonymTable.load(project / "config" / "synonyms.yaml")
    assert table.normalize("我是想要进校访问") == "我是想要入校访问"
    assert table.normalize("外来人员怎么进") == "访客怎么进"


@pytest.mark.parametrize("text", ["访问", "通行", "办理", "系统", "管理"])
def test_coverage_ignores_ambiguous_words(text: str) -> None:
    """只看多义词时，内容词为空 → 覆盖率 0，不能判成"命中"。"""
    retriever = HybridRetriever.__new__(HybridRetriever)  # 不跑 __init__，只测 _coverage
    retriever._bm25 = _StubBm25()
    chunks = [type("C", (), {"text": f"这是关于{text}的规定"})()]
    coverage, tokens = retriever._coverage(text, chunks)  # type: ignore[arg-type]
    assert tokens == []
    assert coverage == 0.0


def test_coverage_keeps_meaningful_words() -> None:
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._bm25 = _StubBm25()
    chunks = [type("C", (), {"text": "入校 需要 预约"})()]
    coverage, tokens = retriever._coverage("入校 访问 怎么 办理", chunks)  # type: ignore[arg-type]
    # "怎么" 是疑问词、"访问/办理" 是多义词，只剩 "入校" 计入分母
    assert tokens == ["入校"]
    assert coverage == 1.0
