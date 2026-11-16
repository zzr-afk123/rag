"""Chroma 向量库实现（本地持久化）。

设计要点：
- 使用 ``PersistentClient`` 嵌入式模式，不需要单独起 Chroma Server；
  目录由 ``vectorstore.persist_dir`` 指定，Docker 里挂卷即可持久化。
- collection 名带 embedding 指纹（``kb__bge-m3-1024``），换模型自动隔离，
  不会出现维度冲突导致的脏数据。
- Chroma 返回的是**距离**（cosine 距离 = 1 - 余弦相似度），这里统一换算成
  "越大越相关"的 score，上层不需要知道底层是距离还是相似度。
- 写入按 500 条一批，避免超大 batch 撑爆内存。

需核实：Chroma 1.x 的 collection 配置入口是 ``configuration={"hnsw": {"space": "cosine"}}``，
旧版是 ``metadata={"hnsw:space": "cosine"}``；本实现按"新 API → 旧 API → 默认"依次降级，
因此升级/降级 Chroma 版本都能跑。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..exceptions import RagAssistantError
from .base import VectorHit, VectorRecord, VectorStore

logger = logging.getLogger("rag_assistant.vectorstore.chroma")

_BATCH_SIZE = 500


class ChromaVectorStore(VectorStore):
    """基于 chromadb 的持久化实现。"""

    name = "chroma"

    def __init__(self, persist_dir: Path, collection: str, *, recreate: bool = False) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 依赖缺失才触发
            raise ImportError("使用 Chroma 需要安装 chromadb：pip install chromadb") from exc

        if hasattr(chromadb, "PersistentClient"):
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        else:  # pragma: no cover - 兼容 0.3.x 老版本
            self._client = chromadb.Client(
                chromadb.config.Settings(is_persistent=True, persist_directory=str(persist_dir))
            )

        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection
        if recreate:
            self._drop_collection()
        self._collection = self._open_collection(collection)

    # ---- 初始化辅助 ----
    def _drop_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("已删除旧 collection：%s", self._collection_name)
        except Exception:  # noqa: BLE001 - 不存在即忽略
            pass

    def _open_collection(self, name: str):
        """按"新配置 API → 旧 metadata API → 默认"顺序尝试，兼容多版本。"""
        try:
            return self._client.get_or_create_collection(
                name=name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
            )
        except TypeError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("新配置 API 不可用：%s", exc)
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        except TypeError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("metadata 配置 API 不可用：%s", exc)
        return self._client.get_or_create_collection(name=name, embedding_function=None)

    # ---- 接口实现 ----
    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        written = 0
        for start in range(0, len(records), _BATCH_SIZE):
            batch = records[start : start + _BATCH_SIZE]
            self._collection.upsert(
                ids=[record.chunk_id for record in batch],
                embeddings=[list(record.embedding) for record in batch],
                documents=[record.text for record in batch],
                metadatas=[dict(record.metadata) for record in batch],
            )
            written += len(batch)
        return written

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[VectorHit]:
        if top_k <= 0:
            return []
        total = self.count()
        if total == 0:
            return []
        n_results = min(top_k, total)
        try:
            result = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=n_results,
                where=dict(where) if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001 - 空库/过滤无命中在部分版本会抛错
            logger.warning("Chroma 查询失败（按无命中处理）：%s", exc)
            return []

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[VectorHit] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    text=str(documents[index]) if index < len(documents) else "",
                    metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                    score=1.0 - distance,  # cosine 距离 → 相似度
                )
            )
        return hits

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, VectorHit]:
        """按 id 批量取回原文与元数据（引用回查用）。"""
        ids = [str(chunk_id) for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return {}
        hits: dict[str, VectorHit] = {}
        for start in range(0, len(ids), _BATCH_SIZE):
            batch = ids[start : start + _BATCH_SIZE]
            try:
                result = self._collection.get(ids=batch, include=["documents", "metadatas"])
            except Exception as exc:  # noqa: BLE001 - id 不存在或版本差异都按"取不到"处理
                logger.warning("Chroma 按 id 取原文失败（跳过 %d 条）：%s", len(batch), exc)
                continue
            found = result.get("ids") or []
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []
            for index, chunk_id in enumerate(found):
                hits[str(chunk_id)] = VectorHit(
                    chunk_id=str(chunk_id),
                    text=str(documents[index]) if index < len(documents) else "",
                    metadata=dict(metadatas[index] or {}) if index < len(metadatas) else {},
                    score=0.0,
                )
        return hits

    def delete_by_doc(self, doc_ids: Sequence[str]) -> int:
        targets = [str(doc_id) for doc_id in doc_ids if doc_id]
        if not targets:
            return 0
        before = self.count()
        try:
            self._collection.delete(where={"doc_id": {"$in": targets}})
        except Exception as exc:  # noqa: BLE001
            raise RagAssistantError(f"删除向量失败：{exc}") from exc
        after = self.count()
        return max(0, before - after)

    def count(self) -> int:
        return int(self._collection.count())

    def list_doc_ids(self) -> set[str]:
        doc_ids: set[str] = set()
        offset = 0
        while True:
            try:
                page = self._collection.get(include=["metadatas"], limit=_BATCH_SIZE, offset=offset)
            except TypeError:  # pragma: no cover - 极老版本不支持 offset
                page = self._collection.get()
            metadatas = page.get("metadatas") or []
            for metadata in metadatas:
                if metadata and metadata.get("doc_id"):
                    doc_ids.add(str(metadata["doc_id"]))
            if len(metadatas) < _BATCH_SIZE:
                break
            offset += _BATCH_SIZE
        return doc_ids

    def reset(self) -> None:
        self._drop_collection()
        self._collection = self._open_collection(self._collection_name)

    def close(self) -> None:
        # chromadb 的 PersistentClient 没有显式 close，释放引用即可
        self._collection = None  # type: ignore[assignment]
