"""配置加载与校验测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_assistant.config import Settings, build_settings, collect_env_overrides, load_settings
from rag_assistant.exceptions import ConfigError


def test_defaults_from_yaml(base_config: dict[str, Any]) -> None:
    """YAML 默认值能正确落到强类型 dataclass。"""
    settings = build_settings(base_config, env={})
    assert isinstance(settings, Settings)
    assert settings.chunking.max_tokens == 700
    assert settings.chunking.overlap_tokens == 60
    assert settings.retrieval.top_n == 5
    assert isinstance(settings.sync.include_patterns, tuple)
    assert "*.md" in settings.sync.include_patterns
    # 无 Key → 自动进入 mock
    assert settings.llm.is_mock is True


def test_env_override_scalar(base_config: dict[str, Any]) -> None:
    """RAG_<SECTION>__<KEY> 环境变量覆盖，且完成类型转换。"""
    settings = build_settings(base_config, env={"RAG_RETRIEVAL__TOP_N": "8", "RAG_RERANKER__ENABLED": "false"})
    assert settings.retrieval.top_n == 8
    assert isinstance(settings.retrieval.top_n, int)
    assert settings.reranker.enabled is False


def test_api_key_from_env(base_config: dict[str, Any]) -> None:
    """Key 只从环境变量读取，配置文件里不出现明文。"""
    settings = build_settings(base_config, env={"DEEPSEEK_API_KEY": "sk-test-123", "DEEPSEEK_BASE_URL": "https://example.com"})
    assert settings.llm.api_key == "sk-test-123"
    assert settings.llm.base_url == "https://example.com"
    assert settings.llm.is_mock is False


def test_force_mock_switch(base_config: dict[str, Any]) -> None:
    """RAG_FORCE_MOCK=1 一次性把三层都切到 mock。"""
    settings = build_settings(
        base_config,
        env={"DEEPSEEK_API_KEY": "sk-test", "RAG_FORCE_MOCK": "1"},
    )
    assert settings.llm.provider == "mock"
    assert settings.embedding.is_mock is True
    assert settings.reranker.is_mock is True


def test_unknown_key_rejected(base_config: dict[str, Any]) -> None:
    """拼错的配置项直接报错，避免"改了没生效"。"""
    broken = {**base_config, "retrieval": {**base_config["retrieval"], "top_nn": 3}}
    with pytest.raises(ConfigError, match="未知配置项"):
        build_settings(broken, env={})


def test_overlap_validation(base_config: dict[str, Any]) -> None:
    """overlap 必须小于 max_tokens。"""
    broken = {**base_config, "chunking": {**base_config["chunking"], "overlap_tokens": 800}}
    with pytest.raises(ConfigError, match="overlap_tokens"):
        build_settings(broken, env={})


def test_threshold_validation(base_config: dict[str, Any]) -> None:
    """阈值必须在 [0,1]。"""
    broken = {**base_config, "retrieval": {**base_config["retrieval"], "score_threshold": 5}}
    with pytest.raises(ConfigError, match="score_threshold"):
        build_settings(broken, env={})


def test_collection_fingerprint(base_config: dict[str, Any]) -> None:
    """collection 名带 embedding 指纹，换模型不冲突（方案假设 #17）。"""
    settings = build_settings(base_config, env={})
    assert settings.embedding_fingerprint == "bge-m3-1024"
    assert settings.collection_name == "kb__bge-m3-1024"


def test_paths_resolved(settings: Settings) -> None:
    """相对路径按项目根解析。"""
    assert settings.raw_dir == settings.root / "data" / "raw"
    assert settings.manifest_path == settings.root / "data" / "index" / "manifest.json"
    assert settings.log_dir == settings.root / "logs"


def test_load_settings_from_file(project: Path) -> None:
    """从真实 YAML 文件加载（.env 不存在时不应报错）。"""
    settings = load_settings(project / "config" / "settings.yaml", root=project, env={})
    assert settings.root == project
    assert settings.project.name == "rag-assistant"


def test_missing_config_file(tmp_path: Path) -> None:
    """配置文件不存在时给出明确错误。"""
    with pytest.raises(ConfigError, match="配置文件不存在"):
        load_settings(tmp_path / "nope.yaml", root=tmp_path, env={})


def test_collect_env_overrides() -> None:
    """环境变量 → dotted path 的映射规则。"""
    overrides = collect_env_overrides(
        {
            "DEEPSEEK_API_KEY": "sk-1",
            "RAG_RETRIEVAL__SCORE_THRESHOLD": "0.6",
            "RAG_SYNC__INTERVAL_SECONDS": "30",
            "PATH": "ignored",
        }
    )
    assert overrides["llm.api_key"] == "sk-1"
    assert overrides["retrieval.score_threshold"] == "0.6"
    assert overrides["sync.interval_seconds"] == "30"
    assert "PATH" not in overrides
