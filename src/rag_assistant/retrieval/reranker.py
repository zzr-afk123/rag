"""重排（Rerank）。

职责分工：
- 向量 + BM25 + RRF 负责**召回**（宁多勿漏，取 candidate_k 条）；
- Reranker 负责**精排**（query 与 chunk 逐对打分），把真正相关的顶上来，
  并且提供一个**可比较的 0–1 分数**用于"无依据就拒答"的阈值判定。

分数口径（重要）：``bge-reranker-v2-m3`` 输出的是 raw logits（可为负、无上界），
直接拿来比阈值没有意义，因此统一过 sigmoid 归一到 (0,1)。这一层归一化是
阈值能跨文档/跨查询稳定的前提。
"""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..config import RerankerConfig
from ..embeddings.base import resolve_device

logger = logging.getLogger("rag_assistant.retrieval.reranker")


def _sigmoid(value: float) -> float:
    """数值稳定的 sigmoid。"""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _char_bigrams(text: str) -> set[str]:
    """字符二元组集合（中文不依赖分词也能算重叠）。"""
    cleaned = "".join(text.split()).lower()
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _lcs_ratio(query: str, text: str) -> float:
    """query 与 text 的最长公共子串占 query 长度的比例（0–1）。

    用于 mock 重排的次要信号：单靠 bigram 覆盖率容易大量并列，
    最长公共子串能把"整段命中'休学办理流程'"和"只命中'休学'两字"区分开。
    """
    if not query or not text:
        return 0.0
    best = 0
    previous = [0] * (len(text) + 1)
    for char in query:
        current = [0] * (len(text) + 1)
        for index, other in enumerate(text, start=1):
            if char == other:
                current[index] = previous[index - 1] + 1
                if current[index] > best:
                    best = current[index]
        previous = current
    return best / len(query)


class Reranker(ABC):
    """重排接口。"""

    name: str = "base"

    @abstractmethod
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """返回每条候选的 [0,1] 相关度分数（顺序与入参一致）。"""

    @property
    def is_mock(self) -> bool:
        return False


class CrossEncoderReranker(Reranker):
    """基于 bge-reranker-v2-m3 的 CrossEncoder 实现（延迟加载）。"""

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self.name = config.model
        self._model: Any | None = None
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._config.device)
        return self._device

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        # 同 embeddings/bge.py：HF_ENDPOINT 必须在 import 之前设置
        if (self._config.hf_endpoint or "").strip():
            os.environ["HF_ENDPOINT"] = self._config.hf_endpoint.strip()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "使用 bge-reranker 需要 sentence-transformers：pip install sentence-transformers "
                "（或设置 reranker.provider=mock / enabled=false 关闭重排）"
            ) from exc
        logger.info("加载 Reranker 模型 %s（device=%s）...", self._config.model, self.device)
        self._model = CrossEncoder(
            self._config.model,
            device=self.device,
            max_length=self._config.max_length,
        )
        if str(self.device).startswith("cuda"):
            # fp16 让权重与激活都减半，8GB 卡上和本地大模型共存才不会被顶到换页
            try:
                self._model.model.half()
                logger.info("Reranker 使用 fp16 推理")
            except Exception as exc:  # pragma: no cover - 个别模型不支持半精度
                logger.warning("Reranker 转 fp16 失败，保持 fp32：%s", exc)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._ensure_model()
        pairs = [(query, text) for text in texts]
        raw = model.predict(
            pairs,
            batch_size=max(1, int(self._config.batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [_sigmoid(float(value)) for value in raw]


class MockReranker(Reranker):
    """无依赖的确定性重排：字符二元组覆盖率 + 最长公共子串。

    它不是语义模型，但能稳定地把"字面更接近 query"的 chunk 排前面，
    足以验证接线、阈值与拒答逻辑（语义质量请用真实模型或看评测报告）。

    注意：它的分数分布与 bge-reranker **不可比**。mock 模式下看到"证据不足"
    不代表检索错了，只说明该分数口径没有为真实阈值标定过。
    """

    name = "mock-lexical"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        query_grams = _char_bigrams(query)
        if not query_grams:
            return [0.0] * len(texts)
        cleaned_query = "".join(query.split()).lower()
        scores: list[float] = []
        for text in texts:
            text_clean = "".join(text.split()).lower()
            overlap = len(query_grams & _char_bigrams(text)) / len(query_grams)
            if len(cleaned_query) >= 4 and cleaned_query in text_clean:
                overlap = max(overlap, 0.9)
            lcs = _lcs_ratio(cleaned_query, text_clean)
            scores.append(round(min(1.0, 0.7 * overlap + 0.3 * lcs), 6))
        return scores

    @property
    def is_mock(self) -> bool:
        return True


def get_reranker(config: RerankerConfig) -> Reranker | None:
    """按配置返回重排器；关闭时返回 None（检索链路自动跳过该步）。"""
    if not config.is_active:
        return None
    provider = config.provider.strip().lower()
    if provider in {"mock", "lexical"}:
        return MockReranker()
    if provider in {"bge", "cross-encoder", "cross_encoder"}:
        return CrossEncoderReranker(config)
    raise ValueError(f"未知的 reranker.provider：{config.provider}")
