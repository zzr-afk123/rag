"""项目自定义异常。

统一放在这里，便于上层（API 层）按异常类型映射 HTTP 状态码，
也便于测试断言"为什么失败"。
"""

from __future__ import annotations


class RagAssistantError(Exception):
    """所有业务异常的基类。"""


class ConfigError(RagAssistantError):
    """配置缺失、类型错误或取值非法。"""


class UnsupportedFormatError(RagAssistantError):
    """文档扩展名不受支持。"""


class ParseError(RagAssistantError):
    """文档解析失败（加密 PDF、损坏的 docx 等）。"""


class ChunkingError(RagAssistantError):
    """切块阶段失败。"""


class ManifestError(RagAssistantError):
    """清单读写或校验失败。"""


class IndexBusyError(RagAssistantError):
    """索引正在被写入，无法获取读锁（阶段 E 使用）。"""


class LLMError(RagAssistantError):
    """LLM 调用失败（网络、鉴权、超时）。"""


class RetrievalError(RagAssistantError):
    """检索链路异常。"""
