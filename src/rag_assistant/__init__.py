"""rag-assistant: 制度/办事文档 RAG 智能问答助手。

包内模块划分：
- config / logging_setup / tokens / exceptions : 基础设施
- ingestion : 文档解析、结构化切块、增量清单
- embeddings / vectorstore / retrieval : 阶段 B
- generation / session / sync / api : 阶段 C 及以后
"""

__version__ = "0.1.0"
