"""摄入流水线：扫描 → 解析 → 切块（阶段 A）；索引写入由 :class:`~rag_assistant.kb.KnowledgeBase` 负责。

职责边界（刻意划清，避免两个地方都能"写索引"）：
- 本模块**只做无副作用的解析与切块**，产物是 :class:`DocIngestResult` / chunk 列表；
- "写向量库、写 BM25、更新清单"这类有状态、需要保证三者一致的动作，
  统一在 ``rag_assistant.kb.KnowledgeBase.sync()`` 里完成。
  这样测试可以只测切块，不必拉起向量库。

阶段 A 的产物 :class:`DryRunReport` 用于在不写任何索引的前提下检查：
"这套解析 + 切块规则在这批文档上到底切出了什么"，是调参的第一现场。

命令行用法::

    python -m rag_assistant.ingestion.pipeline --dry-run            # 全量试算
    python -m rag_assistant.ingestion.pipeline --dry-run --limit 2  # 只看前 2 个文件
    python -m rag_assistant.ingestion.pipeline --dry-run --show-chunks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ..config import Settings, get_settings
from ..exceptions import ParseError, RagAssistantError, UnsupportedFormatError
from ..logging_setup import get_logger, setup_logging
from .chunker import Chunker, ChunkingStats
from .manifest import FileInfo, diff_manifest, load_manifest, scan_directory
from .models import Chunk, ParsedDoc
from .parsers import get_parser

logger = get_logger("ingestion.pipeline")


@dataclass
class DocIngestResult:
    """单个文件的处理结果（成功或失败都记录，便于汇报）。"""

    rel_path: str
    doc: ParsedDoc | None = None
    chunks: list[Chunk] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def tokens(self) -> int:
        return sum(c.tokens for c in self.chunks)


@dataclass
class DryRunReport:
    """试算报告。"""

    results: list[DocIngestResult] = field(default_factory=list)
    stats: ChunkingStats = field(default_factory=ChunkingStats)

    @property
    def ok_results(self) -> list[DocIngestResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[DocIngestResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict[str, object]:
        return {
            "docs": len(self.results),
            "ok": len(self.ok_results),
            "failed": [{"file": r.rel_path, "error": r.error} for r in self.failed],
            "stats": {
                "documents": self.stats.docs,
                "sections": self.stats.sections,
                "chunks": self.stats.chunks,
                "table_chunks": self.stats.table_chunks,
                "tokens": self.stats.tokens,
            },
            "files": [
                {
                    "file": r.rel_path,
                    "sections": len(r.doc.sections) if r.doc else 0,
                    "chunks": len(r.chunks),
                    "tokens": r.tokens,
                    "tables": sum(1 for c in r.chunks if c.chunk_type == "table"),
                    "avg_tokens": round(r.tokens / len(r.chunks), 1) if r.chunks else 0,
                }
                for r in self.ok_results
            ],
        }

    def render(self) -> str:
        """渲染为可读文本表格（等宽对齐，Windows 控制台可直接看）。"""
        lines = ["", "== 试算结果 ==", ""]
        header = f"{'文件':<34}{'章节':>6}{'chunk':>7}{'表格块':>7}{'tokens':>8}{'均长':>7}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.ok_results:
            avg = r.tokens / len(r.chunks) if r.chunks else 0
            name = r.rel_path if len(r.rel_path) <= 32 else "..." + r.rel_path[-29:]
            lines.append(
                f"{name:<34}{len(r.doc.sections) if r.doc else 0:>6}{len(r.chunks):>7}"
                f"{sum(1 for c in r.chunks if c.chunk_type == 'table'):>7}{r.tokens:>8}{avg:>7.1f}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"合计：文档 {self.stats.docs}，章节 {self.stats.sections}，"
            f"chunk {self.stats.chunks}（其中表格块 {self.stats.table_chunks}），"
            f"tokens {self.stats.tokens}"
        )
        if self.failed:
            lines.append("")
            lines.append("失败文件：")
            lines.extend(f"  - {r.rel_path}: {r.error}" for r in self.failed)
        return "\n".join(lines)


class IngestPipeline:
    """文档摄入编排（无副作用）。

    索引写入请调用 ``KnowledgeBase.sync()``；本类不持有向量库与 BM25，
    因此这里的方法都是可重复调用、可单测的纯流程。
    """

    def __init__(self, settings: Settings | None = None, chunker: Chunker | None = None) -> None:
        self.settings = settings or get_settings()
        self.chunker = chunker or Chunker(self.settings.chunking)

    # ---- 扫描 ----
    def scan(self) -> dict[str, FileInfo]:
        """扫描知识库目录。"""
        return scan_directory(
            self.settings.raw_dir,
            include_patterns=self.settings.sync.include_patterns,
            exclude_dirs=self.settings.sync.exclude_dirs,
        )

    def plan(self) -> tuple[dict[str, FileInfo], object]:
        """返回 (当前扫描结果, 同步计划)。"""
        current = self.scan()
        manifest = load_manifest(self.settings.manifest_path)
        return current, diff_manifest(manifest, current)

    # ---- 单文件 ----
    def parse_file(self, info: FileInfo, *, doc_type: str | None = None) -> ParsedDoc:
        """解析单个文件为 :class:`ParsedDoc`。"""
        parser = get_parser(info.abs_path)
        return parser.parse(
            Path(info.abs_path),
            root=self.settings.raw_dir,
            doc_type=doc_type or self.settings.sync.default_doc_type,
        )

    def chunks_for(self, info: FileInfo, *, doc_type: str | None = None) -> list[Chunk]:
        """解析 + 切块单个文件。"""
        doc = self.parse_file(info, doc_type=doc_type)
        return self.chunker.chunk(doc)

    # ---- 批处理 ----
    def iter_results(
        self,
        infos: Sequence[FileInfo] | None = None,
        *,
        limit: int | None = None,
    ) -> Iterator[DocIngestResult]:
        """逐个处理文件，异常按文件隔离（一个坏文件不拖垮整批）。"""
        targets = list(infos if infos is not None else self.scan().values())
        if limit is not None:
            targets = targets[:limit]
        for info in targets:
            try:
                doc = self.parse_file(info)
                chunks = self.chunker.chunk(doc)
                logger.info(
                    "解析完成 %s：%d 章节 / %d chunk / %d tokens",
                    info.rel_path,
                    len(doc.sections),
                    len(chunks),
                    sum(c.tokens for c in chunks),
                )
                yield DocIngestResult(rel_path=info.rel_path, doc=doc, chunks=chunks)
            except (ParseError, UnsupportedFormatError) as exc:
                logger.error("解析失败 %s：%s", info.rel_path, exc)
                yield DocIngestResult(rel_path=info.rel_path, error=str(exc))
            except RagAssistantError as exc:
                logger.error("处理失败 %s：%s", info.rel_path, exc)
                yield DocIngestResult(rel_path=info.rel_path, error=str(exc))
            except Exception as exc:  # noqa: BLE001 - 兜底：未知异常也要记录而不是中断
                logger.exception("未预期异常 %s", info.rel_path)
                yield DocIngestResult(rel_path=info.rel_path, error=f"未预期异常：{exc}")

    def dry_run(self, *, limit: int | None = None) -> DryRunReport:
        """试算：解析 + 切块，不写任何文件。"""
        report = DryRunReport()
        for result in self.iter_results(limit=limit):
            report.results.append(result)
            if result.ok:
                report.stats = report.stats.merge(
                    ChunkingStats(
                        docs=1,
                        sections=len(result.doc.sections) if result.doc else 0,
                        chunks=len(result.chunks),
                        table_chunks=sum(1 for c in result.chunks if c.chunk_type == "table"),
                        tokens=result.tokens,
                    )
                )
        return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="文档摄入试算（不写索引）")
    parser.add_argument("--dry-run", action="store_true", help="试算模式（当前仅支持该模式）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件")
    parser.add_argument("--show-chunks", action="store_true", help="打印每个 chunk 的摘要")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    args = _build_argparser().parse_args(argv)
    setup_logging()
    pipeline = IngestPipeline()
    report = pipeline.dry_run(limit=args.limit)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.render())

    if args.show_chunks:
        for result in report.ok_results:
            print(f"\n--- {result.rel_path} ---")
            for chunk in result.chunks:
                preview = chunk.text[:60].replace("\n", " ")
                print(
                    f"  [{chunk.order:>3}] {chunk.chunk_type:<5} {chunk.tokens:>4}t "
                    f"{chunk.section_path[:40]:<40} | {preview}"
                )

    # 存在解析失败时返回非 0，便于脚本/CI 发现
    return 1 if report.failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
