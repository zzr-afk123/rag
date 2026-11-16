"""bge-m3 Embedding（真实模型）。

依赖 ``sentence-transformers``（会连带安装 torch，体积较大）。**延迟加载**：
只有真正调用时才加载模型，因此 mock 模式、单元测试、`--dry-run` 都不会碰它。

关于指令前缀：bge-m3 是多语种通用模型，**不需要** bge-v1.5 系列那种
"为这个句子生成表示以用于检索中文文档："前缀，加了反而会掉点。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

from ..config import EmbeddingConfig
from .base import Embedder, resolve_device

logger = logging.getLogger("rag_assistant.embeddings.bge")


class BgeEmbedder(Embedder):
    """基于 sentence-transformers 的 bge-m3 实现。"""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self.name = config.model
        self.dim = config.dim
        self._model: Any | None = None
        self._device: str | None = None

    # ---- 模型加载 ----
    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._config.device)
        return self._device

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        # 必须在 import sentence_transformers 之前设置：huggingface_hub 在导入时
        # 读取 HF_ENDPOINT，导入后再改无效。国内网络下 huggingface.co 不可达，
        # 加载模型会先去它那里查元数据，重试五次能卡住一两分钟。
        if (self._config.hf_endpoint or "").strip():
            os.environ["HF_ENDPOINT"] = self._config.hf_endpoint.strip()
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - 依赖缺失才触发
            raise ImportError(
                "使用 bge-m3 需要 sentence-transformers：pip install sentence-transformers "
                "（或无网络时设置 RAG_FORCE_MOCK=1 走 mock 模式）"
            ) from exc

        logger.info("加载 Embedding 模型 %s（device=%s）...", self._config.model, self.device)
        self._model = SentenceTransformer(self._config.model, device=self.device)
        # 允许通过配置裁剪最大长度，降低 CPU 下的编码耗时
        max_seq = getattr(self._model, "max_seq_length", None)
        if max_seq and self._config.max_length and self._config.max_length < max_seq:
            self._model.max_seq_length = self._config.max_length
        actual_dim = int(self._model.get_sentence_embedding_dimension())
        if actual_dim != self.dim:
            logger.warning(
                "配置 embedding.dim=%s 与模型实际维度 %s 不一致，以模型为准",
                self.dim,
                actual_dim,
            )
            self.dim = actual_dim
        return self._model

    # ---- 编码 ----
    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        if not texts:
            return []
        vectors = model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([text])
        return vectors[0] if vectors else [0.0] * self.dim
