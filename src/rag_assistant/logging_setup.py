"""统一日志配置。

要点：
- 读取 ``config/logging.yaml``（标准 dictConfig 格式），加载前替换
  ``${LOG_DIR}`` / ``${LEVEL}`` / ``${RETAIN_DAYS}`` 占位符。
- 文件日志按天轮转（``TimedRotatingFileHandler``，midnight），保留
  ``logging.retain_days`` 天。
- 控制台 handler 做了编码兜底：Windows 中文控制台默认 GBK，日志里出现
  GBK 无法编码的字符时会降级为 ``\\uXXXX``，而不是抛 UnicodeEncodeError
  把主流程打断。
- 每次问答另写一行 JSONL 到 ``logs/qa.jsonl``，供后续统计/评测读取。
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Mapping

import yaml

from .config import Settings, get_settings

_CONFIGURED = False
_QA_LOGGER_NAME = "rag_assistant.qa"


class SafeStreamHandler(logging.StreamHandler):
    """编码不兼容时自动转义的控制台 handler（Windows GBK 控制台友好）。"""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                stream = self.stream
                encoding = getattr(stream, "encoding", None) or "utf-8"
                text = self.format(record)
                stream.write(text.encode(encoding, "backslashreplace").decode(encoding, "replace"))
                stream.write(self.terminator)
                self.flush()
            except Exception:  # noqa: BLE001 - 日志失败绝不冒泡
                self.handleError(record)


def _render_logging_yaml(path: Path, settings: Settings) -> dict[str, Any]:
    """读取 YAML 模板并替换占位符。"""
    template = Template(path.read_text(encoding="utf-8"))
    rendered = template.safe_substitute(
        LOG_DIR=settings.log_dir.as_posix(),
        LEVEL=settings.logging.level.upper(),
        RETAIN_DAYS=str(settings.logging.retain_days),
    )
    config = yaml.safe_load(rendered) or {}
    if not isinstance(config, dict):
        raise ValueError(f"日志配置格式非法：{path}")
    # 用自定义 handler 替换标准 StreamHandler，获得编码兜底能力
    handlers = config.get("handlers", {})
    if isinstance(handlers.get("console"), dict):
        handlers["console"]["()"] = "rag_assistant.logging_setup.SafeStreamHandler"
        handlers["console"].pop("class", None)
    return config


def setup_logging(
    settings: Settings | None = None,
    *,
    config_path: Path | None = None,
    force: bool = False,
) -> None:
    """初始化日志（幂等，可重复调用）。

    Args:
        settings: 传入配置；默认取全局单例。
        config_path: 自定义 logging.yaml 路径（测试用）。
        force: 为 True 时重复调用也会重新应用配置。
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    cfg = settings or get_settings()
    # RAG_LOG_DIR：把"仓库自身的 logs/"重定向到别处（测试用）。
    # 只对落在仓库目录里的日志目录生效——临时项目根（tmp_path）本来就隔离，
    # 那些用例要断言自己目录下的 qa.jsonl，不能一起搬走。
    override = os.environ.get("RAG_LOG_DIR")
    if override:
        from .config import PROJECT_ROOT as _REPO_ROOT

        target = Path(cfg.log_dir).resolve()
        repo = Path(_REPO_ROOT).resolve()
        if target == repo or repo in target.parents:
            cfg = replace(cfg, logging=replace(cfg.logging, dir=override))
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    path = Path(config_path) if config_path else cfg.root / "config" / "logging.yaml"
    if not path.exists():  # pragma: no cover - 兜底配置
        logging.basicConfig(
            level=cfg.logging.level.upper(),
            format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        )
        _CONFIGURED = True
        return

    logging.config.dictConfig(_render_logging_yaml(path, cfg))
    _attach_qa_handler(cfg)
    _quiet_third_party()
    _CONFIGURED = True
    get_logger(__name__).debug("日志系统初始化完成，目录=%s", cfg.log_dir)


#: 这些库的 WARNING 在"模型已下载、离线运行"时没有处置价值，只会把运维台刷满。
_QUIET_LOGGERS = (
    "huggingface_hub",
    "huggingface_hub.hf_api",
    "huggingface_hub.utils._http",
    "urllib3",
    "httpx",
    "filelock",
)


def _quiet_third_party() -> None:
    """把第三方库的噪声日志压到 ERROR。

    只在 yaml 里写 ``loggers.huggingface_hub.level`` 不够稳：子 logger 的
    effective level 由最近一层显式设置 level 的祖先决定，而 ``hf_api`` 在
    import 期自己可能设过 level。这里直接按名字逐个压。
    """
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


def _attach_qa_handler(settings: Settings) -> None:
    """给专用 QA 日志挂一个按天轮转的 JSONL handler（避免重复挂载）。"""
    logger = logging.getLogger(_QA_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    target = settings.log_dir / settings.logging.qa_log
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == target.resolve():
            return
    handler = logging.handlers.TimedRotatingFileHandler(  # type: ignore[attr-defined]
        filename=target,
        when="midnight",
        interval=1,
        backupCount=settings.logging.retain_days,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取带项目命名空间的 logger。"""
    return logging.getLogger(f"rag_assistant.{name}" if not name.startswith("rag_assistant") else name)


def log_qa(record: Mapping[str, Any]) -> None:
    """记录一次问答（JSONL 一行）。

    字段（阶段 C 起逐步补齐）：session_id, question, rewritten_query,
    retrieved_ids, latency_ms, refused, refusal_reason, prompt_tokens,
    completion_tokens。
    """
    logger = logging.getLogger(_QA_LOGGER_NAME)
    if not logger.handlers:  # 未初始化时给出可读提示，但不抛错
        get_logger(__name__).warning("QA 日志 handler 未初始化，记录被丢弃")
        return
    payload = dict(record)
    # 时间戳由这里统一补：文件 handler 的 formatter 只有 %(message)s，
    # 不补的话"近 30 天"只能靠文件 mtime 猜，轮转后精度会很差。
    payload.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def mask_secret(value: str, keep: int = 4) -> str:
    """脱敏输出 Key（日志用）。"""
    if not value:
        return "<empty>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep * 2)}{value[-keep:]}"


# logging.handlers 需要显式导入（dictConfig 字符串引用之外的直接用法）
import logging.handlers  # noqa: E402  (放末尾避免与上面的 import 顺序混乱)

if __name__ == "__main__":  # pragma: no cover - 手动调试点
    setup_logging(force=True)
    get_logger("demo").info("中文日志测试：制度问答助手启动成功")
    log_qa({"session_id": "demo", "question": "在读证明怎么办？", "latency_ms": 12.3, "refused": False})
    print("日志已写入", get_settings().log_dir)
