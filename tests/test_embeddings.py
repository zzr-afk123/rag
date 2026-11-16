"""Embedding 实现测试（mock 实现 + 工厂）。"""

from __future__ import annotations

import math
from typing import Any

import pytest

from rag_assistant.config import Settings, build_settings
from rag_assistant.embeddings import HashEmbedder, get_embedder
from rag_assistant.embeddings.base import resolve_device


def test_hash_embedder_shape_and_norm() -> None:
    """维度正确且已 L2 归一（余弦检索才有稳定口径）。"""
    embedder = HashEmbedder(dim=128)
    vector = embedder.embed_query("学生在读证明怎么开")
    assert len(vector) == 128
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-9


def test_hash_embedder_deterministic() -> None:
    """跨实例、跨调用完全一致（不依赖进程随机盐）。"""
    first = HashEmbedder(dim=64).embed_query("休学办理流程")
    second = HashEmbedder(dim=64).embed_query("休学办理流程")
    assert first == second


def test_hash_embedder_different_texts_differ() -> None:
    """不同文本 → 不同向量；相近文本相似度更高。"""
    embedder = HashEmbedder(dim=256)
    a = embedder.embed_query("休学办理流程")
    b = embedder.embed_query("复学申请材料")
    c = embedder.embed_query("休学办理需要什么流程")
    assert a != b

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y))

    assert cosine(a, c) > cosine(a, b)


def test_hash_embedder_batch() -> None:
    """批量编码与逐条编码结果一致。"""
    embedder = HashEmbedder(dim=32)
    texts = ["第一条 总则", "第二条 范围", ""]
    batch = embedder.embed_documents(texts)
    assert len(batch) == 3
    assert batch[2] == [0.0] * 32
    assert batch[0] == embedder.embed_query(texts[0])


def test_invalid_dim() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_factory_mock(mock_settings: Settings) -> None:
    """provider=mock 时工厂返回哈希实现，并标记 is_mock。"""
    embedder = get_embedder(mock_settings)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.is_mock is True
    assert embedder.dim == mock_settings.embedding.dim


def test_factory_bge_lazy(base_config: dict[str, Any], project: Any) -> None:
    """provider=bge 时返回 BgeEmbedder，但**不会**在构造期加载模型。"""
    from rag_assistant.embeddings import BgeEmbedder

    settings = build_settings(base_config, root=project, env={})
    embedder = get_embedder(settings)
    assert isinstance(embedder, BgeEmbedder)
    assert embedder._model is None  # 延迟加载：构造不触发下载
    assert embedder.is_mock is False


def test_factory_unknown_provider(mock_settings: Settings) -> None:
    with pytest.raises(ValueError):
        get_embedder(
            build_settings(
                {"embedding": {"provider": "not-exist"}},
                root=mock_settings.root,
                env={},
            )
        )


@pytest.mark.parametrize("preference", ["auto", "cpu", "cuda"])
def test_resolve_device_never_crashes(preference: str) -> None:
    """无 torch 环境下也要能解析出可用设备。"""
    assert resolve_device(preference) in {"cpu", "cuda", "mps"}
