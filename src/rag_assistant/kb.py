"""知识库门面（KnowledgeBase）。

把"摄入 + 向量库 + BM25 + 检索"组装成一个对象，供 API 层与 CLI 调用。
职责边界：

- **负责一致性**：向量库、BM25、manifest 三者必须同时更新；
- **不负责业务语义**：指代消解、生成、拒答话术在 generation 层。

## 并发模型（阶段 E）

```
查询：   IndexLock.read()  → 向量检索 + BM25 → 释放
同步：   准备阶段（解析/切块/嵌入）**不持锁**  → 每 N 个文件进一次 IndexLock.write() 提交
         最后在写锁内落 manifest 与 BM25
```

为什么把嵌入放到锁外：嵌入是同步流程里最慢的一步（CPU 上尤其明显），
把它放进写锁会让查询在整个同步期间排队。现在写锁只覆盖"写向量库/写 BM25/写清单"
这些毫秒级动作，查询最多等一个批次。

为什么仍然需要写锁：提交一批时，"删旧 chunk → 写新 chunk"必须原子，
否则并发查询可能读到"旧的删了、新的还没写"的空窗。

## 失败语义

- **单文件失败**：只跳过该文件，清单不更新（下次重试），其余照常提交；
- **提交/落盘阶段的严重异常**：把 manifest 回滚到同步前备份，并记录 ERROR。
  已写入的向量不回滚——因为 chunk_id 是内容哈希派生的，重跑同步是幂等 upsert，
  不会产生重复数据。这是"清单回滚 + 幂等重试"，不是数据库事务。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .exceptions import RagAssistantError
from .ingestion.manifest import (
    FileInfo,
    Manifest,
    apply_plan,
    diff_manifest,
    load_manifest,
    save_manifest,
    scan_directory,
)
from .ingestion.models import Chunk
from .ingestion.pipeline import IngestPipeline
from .ingestion.versions import drop_superseded
from .logging_setup import get_logger
from .retrieval.bm25 import BM25Index, BM25Record, ChineseTokenizer
from .retrieval.models import RetrievalResult
from .retrieval.reranker import Reranker, get_reranker
from .retrieval.retriever import HybridRetriever
from .sync.lock import FileLock, IndexLock, LockOptions
from .vectorstore.base import VectorRecord, VectorStore

logger = get_logger("kb")


@dataclass
class SyncReport:
    """一次同步的结果。"""

    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    indexed_docs: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0
    rebuild: bool = False
    lock_wait_ms: float = 0.0
    superseded: int = 0
    superseded_docs: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "modified": self.modified,
            "removed": self.removed,
            "unchanged": self.unchanged,
            "indexed_docs": self.indexed_docs,
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "failed": self.failed,
            "duration_ms": round(self.duration_ms, 1),
            "rebuild": self.rebuild,
            "lock_wait_ms": round(self.lock_wait_ms, 1),
            "superseded": self.superseded,
            "superseded_docs": self.superseded_docs,
        }

    def summary(self) -> str:
        parts = [
            f"新增 {self.added}",
            f"变更 {self.modified}",
            f"删除 {self.removed}",
            f"未变 {self.unchanged}",
            f"写入 chunk {self.chunks_written}",
            f"耗时 {self.duration_ms:.0f}ms",
        ]
        if self.superseded:
            parts.append(f"旧版本跳过 {self.superseded}")
        if self.failed:
            parts.append(f"失败 {len(self.failed)}")
        return " / ".join(parts)


@dataclass(frozen=True)
class _Prepared:
    """一个文件"已解析、已嵌入、等待提交"的中间产物。"""

    info: FileInfo
    chunks: list[Chunk]
    embeddings: list[list[float]]


class KnowledgeBase:
    """知识库门面对象。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        pipeline: IngestPipeline | None = None,
        lock: IndexLock | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder(self.settings)
        self.vector_store = vector_store or get_vector_store_factory(self.settings)
        self.bm25 = bm25 or self._build_bm25()
        self.reranker = reranker if reranker is not None else get_reranker(self.settings.reranker)
        self.pipeline = pipeline or IngestPipeline(self.settings)
        self.lock = lock or self._build_lock()
        # 整段同步任务的跨进程互斥：进程内已有 SyncJobManager 单飞，但 CLI 与 API
        # 是两个进程，各自算出的 plan 会互相覆盖（清单说 6 个文档、库里只剩 3 个）。
        self._task_lock = FileLock(
            self.settings.resolve(self.settings.sync.lock_file).with_name(".sync-task.lock"),
            stale_seconds=self.settings.sync.stale_lock_s,
        )
        self.retriever = HybridRetriever(
            self.settings,
            self.embedder,
            self.vector_store,
            self.bm25,
            self.reranker,
            lock=self.lock,
        )

    # ---- 构造辅助 ----
    def _build_lock(self) -> IndexLock:
        sync = self.settings.sync
        return IndexLock(
            self.settings.resolve(sync.lock_file),
            LockOptions(
                write_timeout=sync.write_lock_timeout_s,
                read_timeout=sync.read_lock_timeout_s,
                stale_seconds=sync.stale_lock_s,
            ),
        )

    def _build_bm25(self) -> BM25Index:
        tokenizer = ChineseTokenizer(
            userdict_path=self.settings.resolve(self.settings.retrieval.userdict_path),
            stopwords_path=self.settings.resolve(self.settings.retrieval.stopwords_path),
        )
        index = BM25Index(tokenizer, storage_path=self.bm25_path)
        loaded = index.load()
        if loaded:
            logger.info("已从磁盘加载 BM25 索引：%d 条", loaded)
        return index

    @property
    def bm25_path(self) -> Path:
        """BM25 索引文件按 embedding 指纹隔离，与向量 collection 保持一致。"""
        return self.settings.index_dir / "bm25" / f"bm25__{self.settings.embedding_fingerprint}.json"

    # ---- 检索 ----
    def search(
        self,
        query: str,
        *,
        top_n: int | None = None,
        extra_queries: Sequence[str] = (),
        threshold: float | None = None,
    ) -> RetrievalResult:
        """检索入口（内部持有读锁）。"""
        return self.retriever.retrieve(
            query, top_n=top_n, extra_queries=extra_queries, threshold=threshold
        )

    # ---- 原文回查 ----
    def get_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        """按 chunk_id 取回原文与溯源信息（供前端"点引用看原文"）。

        检索结果只带 120 字的引用摘录；要证明"这句话确实出自这里"，
        需要把被引用 chunk 的完整正文取出来。找不到的 id 直接跳过。
        """
        wanted = [str(chunk_id) for chunk_id in chunk_ids if chunk_id]
        if not wanted:
            return []
        found = self.vector_store.get_many(wanted)
        payload: list[dict[str, Any]] = []
        for chunk_id in wanted:
            hit = found.get(chunk_id)
            if hit is None:
                continue
            meta = hit.metadata or {}
            payload.append(
                {
                    "chunk_id": hit.chunk_id,
                    "doc_id": str(meta.get("doc_id", "")),
                    "doc_name": str(meta.get("doc_name", "")),
                    "doc_type": str(meta.get("doc_type", "")),
                    "section_path": str(meta.get("section_path", "")),
                    "page": int(meta.get("page", 0) or 0),
                    "file_version": str(meta.get("file_version", "")),
                    "text": hit.text,
                }
            )
        return payload

    # ---- 状态 ----
    def status(self) -> dict[str, Any]:
        """返回索引概况，供 ``GET /kb/status`` 使用。"""
        manifest = load_manifest(self.settings.manifest_path)
        return {
            "documents": manifest.doc_count,
            "manifest_chunks": manifest.chunk_count,
            "vector_chunks": self.vector_store.count(),
            "bm25_chunks": self.bm25.count,
            "last_sync": manifest.updated_at or None,
            "collection": self.settings.collection_name,
            "embedder": self.embedder.name,
            "embedder_mock": self.embedder.is_mock,
            "reranker": self.reranker.name if self.reranker else None,
            "score_threshold": self.settings.retrieval.score_threshold,
            "raw_dir": str(self.settings.raw_dir),
            "bm25": self.bm25.statistics(),
            "lock": self.lock.status(),
        }

    # ---- 同步 ----
    def sync(
        self,
        *,
        rebuild: bool = False,
        limit: int | None = None,
    ) -> SyncReport:
        """增量同步知识库（解析/嵌入在锁外，提交在写锁内）。

        整段同步持有跨进程的任务锁：同一时刻只允许一个同步任务，否则两个进程会
        各自基于旧清单算 plan，后落盘的那个把对方的成果覆盖掉。

        Args:
            rebuild: True 时清空向量库与 BM25 后全量重建（清单也重置）。
            limit: 只处理前 N 个待同步文件（调试用）。

        Returns:
            :class:`SyncReport`，含每个失败文件的原因与锁等待时间。
        """
        with self._task_lock.hold(timeout=0.0):
            return self._sync_unlocked(rebuild=rebuild, limit=limit)

    def _sync_unlocked(
        self,
        *,
        rebuild: bool = False,
        limit: int | None = None,
    ) -> SyncReport:
        started = time.perf_counter()
        report = SyncReport(rebuild=rebuild)

        scanned = scan_directory(
            self.settings.raw_dir,
            include_patterns=self.settings.sync.include_patterns,
            exclude_dirs=self.settings.sync.exclude_dirs,
        )

        # BM25 索引文件损坏/丢失时会被当成空索引，而文件本身没变，diff 永远是"未变"，
        # 结果关键词召回一直空着。这里做一次自愈：清单里有文档但 BM25 为空就强制重建。
        if not rebuild and scanned and self.bm25.count == 0:
            manifest = load_manifest(self.settings.manifest_path)
            if manifest.docs:
                rebuild = True
                report.rebuild = True
                logger.warning(
                    "BM25 索引为空但清单里有 %d 篇文档，本次按全量重建处理",
                    len(manifest.docs),
                )

        # 同一份文件的多个年份/版本：只索引最新的，旧的当"已删除"处理
        current = scanned
        if getattr(self.settings.sync, "prefer_latest_version", True):
            current, superseded = drop_superseded(scanned)
            if superseded:
                report.superseded = len(superseded)
                for old_id, keep_id in sorted(superseded.items()):
                    old = scanned.get(old_id)
                    keep = scanned.get(keep_id)
                    if old and keep:
                        report.superseded_docs.append(
                            {"file": old.rel_path, "superseded_by": keep.rel_path}
                        )
                        logger.info(
                            "版本去重：%s 已被 %s 取代，只索引最新版本",
                            old.rel_path,
                            keep.rel_path,
                        )

        if rebuild:
            wait_started = time.perf_counter()
            with self.lock.write():
                logger.info("全量重建：清空向量库与 BM25 索引")
                self.vector_store.reset()
                self.bm25.clear()
            report.lock_wait_ms += (time.perf_counter() - wait_started) * 1000
            manifest = Manifest()
            plan = diff_manifest(manifest, current)
        else:
            manifest = load_manifest(self.settings.manifest_path)
            plan = diff_manifest(manifest, current)

        report.added = len(plan.added)
        report.modified = len(plan.modified)
        report.removed = len(plan.removed)
        report.unchanged = len(plan.unchanged)

        backup = self._backup_manifest()
        chunk_counts: dict[str, int] = {}
        modified_ids = {info.doc_id for info in plan.modified}

        try:
            # 1) 删除已消失的文档
            if plan.removed:
                wait_started = time.perf_counter()
                with self.lock.write():
                    report.chunks_deleted += self.vector_store.delete_by_doc(plan.removed)
                    self.bm25.remove_docs(plan.removed)
                report.lock_wait_ms += (time.perf_counter() - wait_started) * 1000
                logger.info("已删除 %d 个文档的索引", len(plan.removed))

            # 2) 准备（解析 + 切块 + 嵌入）——不持锁
            touched = plan.touched[: limit if limit else None]
            prepared: list[_Prepared] = []
            for info in touched:
                # 心跳：解析大文件/嵌入耗时可能超过 stale_lock_s，锁被别的进程
                # 判为"崩溃残留"清掉，就会出现两个同步同时写索引
                self._task_lock.touch()
                item, error = self._prepare(info)
                if item is None:
                    report.failed.append(
                        {"file": info.rel_path, "stage": "prepare", "error": error or "解析/嵌入失败"}
                    )
                    continue
                prepared.append(item)
                if len(prepared) >= self.settings.sync.commit_batch_docs:
                    self._commit_batch(prepared, modified_ids, report, chunk_counts)
                    prepared.clear()
            if prepared:
                self._commit_batch(prepared, modified_ids, report, chunk_counts)
        except Exception as exc:  # noqa: BLE001 - 严重错误：回滚清单并结束
            logger.exception("同步过程发生严重错误，回滚清单")
            report.failed.append({"file": "<sync>", "stage": "commit", "error": f"{type(exc).__name__}: {exc}"})
            self._restore_manifest(backup)
            report.duration_ms = (time.perf_counter() - started) * 1000
            return report

        # 3) 落盘清单与 BM25（写锁内，保证与向量写入的可见性一致）
        try:
            wait_started = time.perf_counter()
            with self.lock.write():
                updated = apply_plan(manifest, plan, chunk_counts)
                save_manifest(self.settings.manifest_path, updated)
                self.bm25.save()
            report.lock_wait_ms += (time.perf_counter() - wait_started) * 1000
        except Exception as exc:  # noqa: BLE001
            logger.exception("清单/BM25 落盘失败")
            report.failed.append({"file": "<manifest>", "stage": "persist", "error": str(exc)})
            self._restore_manifest(backup)

        report.duration_ms = (time.perf_counter() - started) * 1000
        logger.info("同步完成：%s（锁等待 %.0fms）", report.summary(), report.lock_wait_ms)
        return report

    # ---- 同步内部 ----
    def _prepare(self, info: FileInfo) -> tuple[_Prepared | None, str | None]:
        """解析 + 切块 + 嵌入（不持锁）。失败时返回 (None, 错误信息)。"""
        try:
            doc = self.pipeline.parse_file(info)
            chunks = self.pipeline.chunker.chunk(doc)
            if not chunks:
                raise RagAssistantError("切块结果为空，跳过该文件")
            embeddings = self._embed_chunks([chunk.embed_text for chunk in chunks])
            return _Prepared(info=info, chunks=chunks, embeddings=embeddings), None
        except Exception as exc:  # noqa: BLE001 - 单文件失败隔离
            logger.exception("准备失败：%s", info.rel_path)
            return None, f"{type(exc).__name__}: {exc}"

    def _commit_batch(
        self,
        batch: Sequence[_Prepared],
        modified_ids: set[str],
        report: SyncReport,
        chunk_counts: dict[str, int],
    ) -> None:
        """在写锁内提交一批文件（删旧 + 写新 + 更新 BM25）。"""
        wait_started = time.perf_counter()
        with self.lock.write():
            report.lock_wait_ms += (time.perf_counter() - wait_started) * 1000
            for item in batch:
                try:
                    if item.info.doc_id in modified_ids:
                        report.chunks_deleted += self.vector_store.delete_by_doc([item.info.doc_id])
                        self.bm25.remove_docs([item.info.doc_id])

                    records = [
                        VectorRecord(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            metadata=chunk.to_metadata(),
                            embedding=embedding,
                        )
                        for chunk, embedding in zip(item.chunks, item.embeddings)
                    ]
                    written = self.vector_store.upsert(records)
                    self.bm25.add(
                        [
                            BM25Record(
                                chunk_id=chunk.chunk_id,
                                text=chunk.text,
                                metadata=chunk.to_metadata(),
                            )
                            for chunk in item.chunks
                        ]
                    )
                    chunk_counts[item.info.doc_id] = len(item.chunks)
                    report.chunks_written += written
                    report.indexed_docs += 1
                    logger.info("已入库 %s：%d chunk", item.info.rel_path, len(item.chunks))
                except Exception as exc:  # noqa: BLE001 - 单文件提交失败不影响同批其他文件
                    logger.exception("提交失败：%s", item.info.rel_path)
                    report.failed.append(
                        {"file": item.info.rel_path, "stage": "commit", "error": f"{type(exc).__name__}: {exc}"}
                    )

    def _backup_manifest(self) -> Path | None:
        """同步前备份清单，供严重错误时回滚。"""
        source = self.settings.manifest_path
        if not source.exists():
            return None
        backup = source.with_suffix(source.suffix + ".bak")
        try:
            shutil.copy2(source, backup)
            return backup
        except OSError:  # pragma: no cover - 权限问题
            logger.warning("清单备份失败，将无法回滚")
            return None

    def _restore_manifest(self, backup: Path | None) -> None:
        if backup is None or not backup.exists():
            return
        try:
            shutil.copy2(backup, self.settings.manifest_path)
            logger.warning("已把清单回滚到本次同步前的状态：%s", self.settings.manifest_path)
        except OSError:  # pragma: no cover
            logger.exception("清单回滚失败，需人工检查")

    def _embed_chunks(self, texts: Sequence[str]) -> list[list[float]]:
        """分批编码，避免一次吃进过多显存/内存。"""
        if not texts:
            return []
        group = max(1, self.settings.embedding.batch_size * 4)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), group):
            vectors.extend(self.embedder.embed_documents(texts[start : start + group]))
        if len(vectors) != len(texts):
            raise RagAssistantError(f"embedding 数量与 chunk 数量不一致：{len(vectors)} != {len(texts)}")
        return vectors

    # ---- 重建 ----
    def reset(self) -> None:
        """清空所有索引与清单（写锁内执行）。"""
        with self.lock.write():
            self.vector_store.reset()
            self.bm25.clear()
            self.bm25.save()
            save_manifest(self.settings.manifest_path, Manifest())
        logger.warning("知识库索引与清单已清空")

    def close(self) -> None:
        """释放底层资源。"""
        self.retriever.close()


def get_vector_store_factory(settings: Settings) -> VectorStore:
    """延迟导入向量库工厂，避免 import 期加载 chromadb。"""
    from .vectorstore import get_vector_store

    return get_vector_store(settings)
