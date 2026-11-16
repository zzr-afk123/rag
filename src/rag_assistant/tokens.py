"""token 计数与文本切分工具。

设计说明（对应方案假设 #6）：
- 优先使用 ``tiktoken`` 的 ``cl100k_base`` 编码做近似计数。DeepSeek 的真实
  tokenizer 未公开，cl100k 对中文的估算是"偏多"，用于控制 chunk 上限是安全
  的方向（宁可切小一点，也不会超模型上下文）。
- 首次使用时 ``tiktoken`` 需要联网下载词表；离线环境下自动降级为
  :func:`heuristic_token_count`，保证 pipeline 与测试在任何环境都能跑通。
- 全模块只用整数 token 估算与句子边界切分，不引入 embedding 依赖。
"""

from __future__ import annotations

import functools
import re
from typing import Final, Protocol

# CJK 统一表意文字（含扩展 A、兼容区）
_CJK_PATTERN: Final = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# 非 CJK 的"词"：连续字母数字
_WORD_PATTERN: Final = re.compile(r"[A-Za-z0-9]+")
# 句子级切分：中文句末标点、英文句点（排除小数点）、分号、换行
_SENTENCE_SPLIT: Final = re.compile(r"[^。！？；\n!?;]*(?:[。！？；\n!?;]+|$)")


class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...  # pragma: no cover


@functools.lru_cache(maxsize=1)
def _get_encoder() -> _Encoder | None:
    """尝试加载 tiktoken 编码器；不可用（未安装/无网络）时返回 None。"""
    try:  # pragma: no cover - 依赖运行环境
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - 任何原因都降级，不影响主流程
        return None


def heuristic_token_count(text: str) -> int:
    """无 tiktoken 时的启发式估算。

    规则：每个 CJK 字符计 1 token；每个英文/数字词按 ``ceil(len/3.5)`` 计；
    其余可见标点各计 1。这是粗估，误差约 ±15%，
    只用于 chunk 长度控制，不用于计费或上报。
    """
    if not text:
        return 0
    cjk = len(_CJK_PATTERN.findall(text))
    rest = _CJK_PATTERN.sub(" ", text)
    words = _WORD_PATTERN.findall(rest)
    word_tokens = sum(max(1, int(len(w) / 3.5 + 0.5)) for w in words)
    punctuation = len(re.findall(r"[^\sA-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", rest))
    return cjk + word_tokens + punctuation


def count_tokens(text: str) -> int:
    """返回文本的估算 token 数。"""
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:  # pragma: no cover - 依赖运行环境
            return len(encoder.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return heuristic_token_count(text)


def split_sentences(text: str) -> list[str]:
    """按中文句末标点/换行切句，保留标点，过滤空白句。

    不会在"第3.5条""1.2 万元"这类小数处断句（只对中文全角句号与英文
    ``! ? ;`` 敏感，英文 ``.`` 需后接空白或结尾才断）。
    """
    if not text:
        return []
    # 先把英文句点规范化：仅在"点 + 空白/结尾"处断开，避免切断小数与缩写
    normalized = re.sub(r"(?<=[A-Za-z\u4e00-\u9fff])\.(?=\s|$)", "。", text)
    parts = [m.group(0).strip() for m in _SENTENCE_SPLIT.finditer(normalized)]
    return [p for p in parts if p]


def tail_tokens(text: str, n_tokens: int) -> str:
    """取文本尾部约 ``n_tokens`` 个 token 的内容，并对齐到句子边界。

    用于 chunk 之间的重叠（overlap），避免把句子切成半句。
    """
    if n_tokens <= 0 or not text:
        return ""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    picked: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        size = count_tokens(sentence)
        if total + size > n_tokens and picked:
            break
        picked.append(sentence)
        total += size
        if total >= n_tokens:
            break
    return "".join(reversed(picked))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """把文本截断到不超过 ``max_tokens``（按比例粗截，用于日志/提示词保护）。"""
    if max_tokens <= 0 or not text:
        return ""
    total = count_tokens(text)
    if total <= max_tokens:
        return text

    cut = max(1, int(len(text) * max_tokens / total))
    # 按比例截断是估算，可能仍略超；逐步收缩直到满足上限
    while cut > 1 and count_tokens(text[:cut]) > max_tokens:
        cut = int(cut * 0.95)
    window = text[:cut]
    # 尽量对齐到句子边界（对齐后不超限才采用）
    sentences = split_sentences(window)
    if sentences:
        aligned = "".join(sentences)
        if aligned and count_tokens(aligned) <= max_tokens:
            return aligned
    return window
