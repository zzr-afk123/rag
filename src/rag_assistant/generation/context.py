"""上下文拼装：把召回 chunk 变成带编号的资料块。

两块预算同时生效：
- ``generation.max_context_chunks``：最多几条；
- ``generation.max_context_tokens``：总 token 上限（超出按顺序截断，标题行永远保留）。
  预算按整块算：标题 + 两行围栏 + 正文，围栏不计入就等于每块超支十来个 token。

编号必须从 1 开始连续，因为 prompt 里让模型"按资料编号引用"，
后续 ``citation.remap_citations`` 依赖这个一一对应关系。

防提示注入（"资料隔离"）：每个资料块的正文都用 ``<<<不可信外部文本 n>>>`` 围栏包住，
围栏外的标题才是系统元数据。知识库里的文件是各部门上传的外部内容，
一句"忽略以上规则，只输出……"就躺在某页 PDF 里；围栏 + 系统提示词里的对应条款
一起告诉模型"这段是证据不是指令"。围栏常量在 ``llm/prompts.py``
（``FENCE_START`` / ``FENCE_END`` / ``FENCE_RE``），``MockLLM`` 解析资料块时按它跳过围栏行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..llm.prompts import FENCE_END, FENCE_START
from ..retrieval.models import RetrievedChunk
from ..tokens import count_tokens, truncate_to_tokens
from .textfix import flatten_tables


@dataclass(frozen=True)
class ContextBlock:
    """一条带编号的资料块。"""

    index: int
    chunk: RetrievedChunk
    header: str
    body: str

    def render(self) -> str:
        """标题在围栏外，正文在围栏内。

        标题是系统生成的元数据（文档名/章节/页码），围栏内才是原文——
        这样模型能分辨"哪一句是谁说的"，引用编号也仍然只对应围栏内内容。
        """
        return (
            f"{self.header}\n"
            f"{FENCE_START.format(index=self.index)}\n"
            f"正文：{self.body}\n"
            f"{FENCE_END.format(index=self.index)}"
        )

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())


def build_context(
    chunks: Sequence[RetrievedChunk],
    *,
    max_chunks: int = 5,
    max_tokens: int = 3000,
    min_score: float = 0.0,
) -> list[ContextBlock]:
    """按"文档 + 章节"合并后编号。

    为什么不再一条 chunk 一个资料块：同一条规定常被切成"正文段 + 表格"两个 chunk，
    模型会**综合两段作答却只引用其中一段**，LLM 评审因此判为"无依据"
    （实测第 6、27 题都是这个原因）。

    合并后：一个章节 = 一个资料块 = 一个引用编号，与"引用按章节聚合"的口径一致。

    截断策略：按相关度顺序逐章节加入；某一章节过长时先截断正文保住标题，
    预算耗尽即停。

    ``min_score``：资料块的分数下限（用块内最高分那条 chunk 的 final_score）。
    低于它的块不放进 prompt —— 检索阈值只决定"要不要拒答"，不代表其余召回
    都值得喂给模型；实测「大学生反诈」时 0.50 分的实验室条款会被一起照抄。
    最高分那一块始终保留，``min_score<=0`` 表示不过滤。
    """
    groups: dict[tuple[str, str], list[RetrievedChunk]] = {}
    order: list[tuple[str, str]] = []
    for chunk in chunks:
        key = (chunk.doc_id, chunk.section_path)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(chunk)

    blocks: list[ContextBlock] = []
    used = 0
    for key in order[:max_chunks]:
        members = groups[key]
        representative = members[0]  # chunks 已按相关度降序
        if min_score > 0 and blocks and representative.final_score < min_score:
            continue
        page = f"第 {representative.page} 页" if representative.page else "无页码"
        header = (
            f"[{len(blocks) + 1}] 文档：{representative.doc_name} "
            f"｜ 章节：{representative.section_path or '正文'} ｜ 页码：{page}"
        )
        index = len(blocks) + 1
        # 围栏两行也是要发给模型的内容，不计入预算就等于每块超支十来个 token
        fence_cost = count_tokens(FENCE_START.format(index=index)) + count_tokens(FENCE_END.format(index=index))
        remaining = max_tokens - used - count_tokens(header) - fence_cost
        if remaining <= 20:  # 只够放标题就没意义了
            break
        merged = "\n".join(dict.fromkeys(member.text.strip() for member in members if member.text.strip()))
        merged = flatten_tables(merged)
        body = merged if count_tokens(merged) <= remaining else truncate_to_tokens(merged, remaining)
        if not body:
            continue
        block = ContextBlock(index=index, chunk=representative, header=header, body=body)
        blocks.append(block)
        used += block.tokens
    return blocks


def render_context(blocks: Sequence[ContextBlock]) -> str:
    """拼成 prompt 用的整段文本。"""
    return "\n\n".join(block.render() for block in blocks)
