"""BM25 关键字检索（中文分词 + 增量维护 + 落盘）。

为什么中文必须分词：BM25 是词袋模型，"教务处"被切成"教/务/处"后与用户 query
的匹配关系会失真。这里用 jieba 的**独立实例**（不用全局 jieba），配合项目自定义
词典（学院术语）与停用词表。

增量维护的取舍：``rank_bm25`` 的 ``BM25Okapi`` 需要在构造时给定完整语料，
无法真正"插入一条"。因此策略是**内存保留语料，增删后惰性标记 dirty，
下次检索时重建**。制度库规模（几千 chunk）下重建耗时 <100ms，可接受；
这条取舍写在代码里，避免后人误以为它是 O(1) 插入。

持久化用 JSON（不用 pickle）：内容可读、可 diff、无反序列化执行风险。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..exceptions import RagAssistantError

logger = logging.getLogger("rag_assistant.retrieval.bm25")

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PUNCT_OR_SPACE = re.compile(r"^[\W_]+$", re.UNICODE)
_STORAGE_VERSION = 1


@dataclass(frozen=True)
class BM25Record:
    """BM25 索引中的一条记录。"""

    chunk_id: str
    text: str
    metadata: Mapping[str, str | int | float]


@dataclass(frozen=True)
class BM25Hit:
    """一条 BM25 命中。"""

    chunk_id: str
    text: str
    metadata: Mapping[str, str | int | float]
    score: float


class ChineseTokenizer:
    """jieba 分词 + 停用词过滤 + 术语归一。

    过滤规则：
    - 纯标点/空白丢弃；
    - 停用词丢弃；
    - 中文单字丢弃（噪声大），ASCII 单词/数字保留（"800"、"PDF" 有信息量）；
    - 全部转小写，便于英文大小写不敏感匹配。
    """

    def __init__(
        self,
        *,
        userdict_path: Path | None = None,
        stopwords_path: Path | None = None,
    ) -> None:
        try:
            import jieba
        except ImportError as exc:  # pragma: no cover
            raise ImportError("中文分词需要 jieba：pip install jieba") from exc

        self._jieba = jieba.Tokenizer()
        # jieba 默认往 stderr 打"Building prefix dict..."，会污染日志与 CLI 输出
        try:
            jieba.setLogLevel(logging.WARNING)
        except Exception:  # noqa: BLE001 - 老版本没有该函数
            pass
        self._stopwords = self._load_stopwords(stopwords_path)
        if userdict_path and Path(userdict_path).exists():
            try:
                self._jieba.load_userdict(str(userdict_path))
                logger.debug("已加载 jieba 自定义词典：%s", userdict_path)
            except Exception as exc:  # noqa: BLE001 - 词典问题不应阻塞检索
                logger.warning("自定义词典加载失败（忽略）：%s", exc)

    @staticmethod
    def _load_stopwords(path: Path | None) -> set[str]:
        if not path or not Path(path).exists():
            return set()
        words: set[str] = set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word.lower())
        return words

    def tokenize(self, text: str) -> list[str]:
        """切成词列表（保留重复，BM25 需要词频）。"""
        if not text:
            return []
        raw = self._jieba.lcut(text.lower())
        tokens: list[str] = []
        for token in raw:
            word = token.strip()
            if not word or _PUNCT_OR_SPACE.match(word):
                continue
            if word in self._stopwords:
                continue
            is_cjk = bool(_CJK.search(word))
            if is_cjk and len(word) < 2:
                continue
            tokens.append(word)
        return tokens


class BM25Index:
    """BM25 倒排索引（内存 + JSON 落盘）。"""

    def __init__(
        self,
        tokenizer: ChineseTokenizer,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        storage_path: Path | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b
        self._storage_path = Path(storage_path) if storage_path else None
        self._records: dict[str, BM25Record] = {}
        self._tokens: dict[str, list[str]] = {}
        self._ids: list[str] = []
        self._bm25 = None
        self._dirty = True

    # ---- 状态 ----
    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def tokenizer(self) -> ChineseTokenizer:
        return self._tokenizer

    def doc_ids(self) -> set[str]:
        return {str(rec.metadata.get("doc_id", "")) for rec in self._records.values() if rec.metadata.get("doc_id")}

    # ---- 写入 ----
    def add(self, records: Sequence[BM25Record]) -> int:
        """新增或覆盖记录（同 chunk_id 覆盖）。"""
        for record in records:
            self._records[record.chunk_id] = record
            self._tokens[record.chunk_id] = self._tokenizer.tokenize(record.text)
            if record.chunk_id not in self._ids:
                self._ids.append(record.chunk_id)
        if records:
            self._dirty = True
        return len(records)

    def remove_docs(self, doc_ids: Iterable[str]) -> int:
        """删除指定文档的所有 chunk。"""
        targets = {str(doc_id) for doc_id in doc_ids}
        if not targets:
            return 0
        removed = [
            chunk_id
            for chunk_id, record in self._records.items()
            if str(record.metadata.get("doc_id", "")) in targets
        ]
        for chunk_id in removed:
            self._records.pop(chunk_id, None)
            self._tokens.pop(chunk_id, None)
        if removed:
            self._ids = [chunk_id for chunk_id in self._ids if chunk_id in self._records]
            self._dirty = True
        return len(removed)

    def clear(self) -> None:
        """清空索引。"""
        self._records.clear()
        self._tokens.clear()
        self._ids.clear()
        self._bm25 = None
        self._dirty = True

    # ---- 检索 ----
    def _rebuild(self) -> None:
        """重建 BM25 索引（惰性，增删后首次检索时触发）。

        用 ``BM25Plus`` 而不是 ``BM25Okapi``：Okapi 的 idf 为
        ``log((N - n + 0.5) / (n + 0.5))``，当某个词出现在**全部**文档中
        （小语料、单文档测试、或"休学"这类高频词）时 idf 为负，
        rank_bm25 会把所有负分裁掉，表现为"明明命中却搜不到"。
        BM25Plus 的 idf 是 ``log((N + 1) / n)``，恒为正，
        并且 delta 项保证单次命中也能出分 —— 对几百到几千 chunk 的制度库更稳。
        """
        if not self._dirty:
            return
        if not self._ids:
            self._bm25 = None
            self._dirty = False
            return
        try:
            from rank_bm25 import BM25Plus
        except ImportError as exc:  # pragma: no cover
            raise ImportError("BM25 需要 rank-bm25：pip install rank-bm25") from exc
        corpus = [self._tokens[chunk_id] for chunk_id in self._ids]
        # 语料里若出现全空 token 列表，BM25 计算平均长度会产生 0 除，补一个占位符
        corpus = [tokens or ["\u0000"] for tokens in corpus]
        self._bm25 = BM25Plus(corpus, k1=self._k1, b=self._b, delta=1.0)
        self._dirty = False

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        """检索，返回按分数降序且分数 > 0 的命中。"""
        if top_k <= 0 or not self._records:
            return []
        query_tokens = self._tokenizer.tokenize(query)
        if not query_tokens:
            return []
        self._rebuild()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = [(self._ids[index], float(score)) for index, score in enumerate(scores) if score > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        hits: list[BM25Hit] = []
        for chunk_id, score in ranked[:top_k]:
            record = self._records[chunk_id]
            hits.append(
                BM25Hit(
                    chunk_id=chunk_id,
                    text=record.text,
                    metadata=record.metadata,
                    score=score,
                )
            )
        return hits

    # ---- 持久化 ----
    def save(self, path: Path | None = None) -> Path:
        """落盘为 JSON。"""
        target = Path(path) if path else self._storage_path
        if target is None:
            raise RagAssistantError("未指定 BM25 索引存储路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _STORAGE_VERSION,
            "k1": self._k1,
            "b": self._b,
            "ids": self._ids,
            "records": {
                chunk_id: {
                    "text": record.text,
                    "metadata": dict(record.metadata),
                    "tokens": self._tokens.get(chunk_id, []),
                }
                for chunk_id, record in self._records.items()
            },
        }
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        logger.debug("BM25 索引已保存：%s（%d 条）", target, len(self._records))
        return target

    def load(self, path: Path | None = None) -> int:
        """从 JSON 恢复，返回加载条数。文件不存在时视为空索引。"""
        target = Path(path) if path else self._storage_path
        if target is None or not target.exists():
            return 0
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("BM25 索引文件损坏，按空索引处理：%s", exc)
            return 0
        if int(payload.get("version", 0)) > _STORAGE_VERSION:
            logger.warning("BM25 索引版本过新（%s），忽略并重建", payload.get("version"))
            return 0

        self.clear()
        self._k1 = float(payload.get("k1", self._k1))
        self._b = float(payload.get("b", self._b))
        records = payload.get("records") or {}
        for chunk_id, item in records.items():
            self._records[chunk_id] = BM25Record(
                chunk_id=chunk_id,
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata") or {}),
            )
            self._tokens[chunk_id] = list(item.get("tokens") or [])
        self._ids = [chunk_id for chunk_id in payload.get("ids", []) if chunk_id in self._records]
        # 兼容 ids 缺失/顺序错乱的旧文件
        if len(self._ids) != len(self._records):
            self._ids = list(self._records)
        self._dirty = True
        return len(self._records)

    def statistics(self) -> dict[str, float | int]:
        """索引概况（供 /kb/status 使用）。"""
        token_total = sum(len(tokens) for tokens in self._tokens.values())
        return {
            "chunks": len(self._records),
            "unique_tokens": len({token for tokens in self._tokens.values() for token in tokens}),
            "avg_tokens": round(token_total / len(self._records), 1) if self._records else 0.0,
        }
