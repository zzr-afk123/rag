"""同步相关的并发原语与自动化。

刻意**不在本 __init__ 里 eager import 子模块**：``sync.scheduler`` 依赖
``api.jobs``→``kb``→``sync.lock``，如果这里把 scheduler 拉进来，会形成
"kb 导入 sync 包 → sync 包导入 scheduler → scheduler 导入 kb" 的循环。
因此请直接按子模块导入：

    from rag_assistant.sync.lock import IndexLock
    from rag_assistant.sync.scheduler import SyncScheduler
    from rag_assistant.sync.watcher import RawDirWatcher
"""
