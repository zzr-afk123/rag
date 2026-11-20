"""配置加载：``config/settings.yaml`` + ``.env`` + 环境变量覆盖。

优先级（后者覆盖前者）：
1. ``config/settings.yaml`` 中声明的默认值
2. ``.env`` 中的专用变量（``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL`` /
   ``RAG_LLM__MODEL`` 等）
3. 任意形如 ``RAG_<SECTION>__<KEY>`` 的环境变量（大小写不敏感）

之所以不用 pydantic-settings：本模块只有"读 YAML + 类型转换 + 校验"三件事，
用标准库 dataclass 即可，能减少一层依赖；FastAPI 的请求/响应模型仍然用
pydantic（阶段 D）。
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigError

T = TypeVar("T")

#: 项目根目录（包含 config/ 与 src/ 的那一层）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


# --------------------------------------------------------------------------
# 各分节配置
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    """项目级元信息。"""

    name: str = "rag-assistant"
    timezone: str = "Asia/Shanghai"


@dataclass(frozen=True)
class LLMConfig:
    """LLM 调用参数（DeepSeek / 本地 OpenAI 兼容服务 / mock）。

    本地服务（LM Studio、Ollama、vLLM 等）都提供 OpenAI 兼容接口，
    因此 ``provider`` 只影响"是否需要 API Key"和默认值，
    真正的调用都是同一条 HTTP 协议。
    """

    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    api_key: str = field(default="", repr=False)
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: int = 60
    max_retries: int = 2
    #: 额外请求参数，原样塞进 chat/completions 的 payload。
    #: 典型用途：让本地思考型模型别再"想"太久，例如
    #: ``model_kwargs: {chat_template_kwargs: {enable_thinking: false}}``
    #: （是否生效取决于推理后端与模型模板，支持不支持都要实测）。
    model_kwargs: dict[str, object] = field(default_factory=dict)
    #: 备用模型：主模型（如云端 DeepSeek）调用失败或未配置 Key 时自动切换。
    #: 典型用法：主 = deepseek，备 = 本地 LM Studio，实现"云端优先、本地兜底"。
    fallback_provider: str = ""
    fallback_base_url: str = ""
    fallback_model: str = ""

    @property
    def has_fallback(self) -> bool:
        return bool(self.fallback_provider.strip() and self.fallback_model.strip())

    @property
    def is_local(self) -> bool:
        """是否指向本地推理服务（不需要 Key）。"""
        return self.provider.strip().lower() in {"openai_compatible", "lmstudio", "local", "ollama", "vllm"}

    @property
    def requires_key(self) -> bool:
        return not self.is_local

    @property
    def is_mock(self) -> bool:
        """是否走 mock：显式 mock，或需要 Key 但没配。"""
        if self.provider.strip().lower() == "mock":
            return True
        if not self.requires_key:
            return False
        return not self.api_key


@dataclass(frozen=True)
class EmbeddingConfig:
    """向量化参数。``device=auto`` 表示有 CUDA 用 GPU，否则 CPU。"""

    provider: str = "bge"
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    device: str = "auto"
    batch_size: int = 16
    normalize: bool = True
    max_length: int = 512
    #: 模型元数据查询/下载走的 HuggingFace 端点（默认镜像站）。
    #: 官方站被墙时，sentence-transformers 加载模型前会去 huggingface.co 查元数据，
    #: 5 次退避重试能卡住一两分钟甚至直接失败，必须先指向镜像。
    hf_endpoint: str = "https://hf-mirror.com"

    @property
    def is_mock(self) -> bool:
        return self.provider.strip().lower() == "mock"


@dataclass(frozen=True)
class VectorStoreConfig:
    """向量库参数。"""

    backend: str = "chroma"
    persist_dir: str = "data/index/chroma"
    collection: str = "kb"


@dataclass(frozen=True)
class RerankerConfig:
    """重排参数。"""

    enabled: bool = True
    provider: str = "bge"
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "auto"
    top_k_in: int = 30
    max_length: int = 512
    # 单批送进模型的 query-段落对数量。8GB 显卡上 32 会把显存顶满，
    # Windows 的 WDDM 一旦开始换页，重排会从 0.6s 掉到几十秒。
    batch_size: int = 8
    hf_endpoint: str = "https://hf-mirror.com"

    @property
    def is_mock(self) -> bool:
        return self.provider.strip().lower() == "mock"

    @property
    def is_active(self) -> bool:
        return self.enabled and self.provider.strip().lower() != "none"


@dataclass(frozen=True)
class RetrievalConfig:
    """混合检索参数。"""

    top_n: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    weight_vector: float = 1.0
    weight_bm25: float = 1.0
    enable_multi_query: bool = True
    multi_query: int = 3
    score_threshold: float = 0.45
    #: 进入 prompt 的资料块最低分：低于它的块不进上下文。
    #: 只按 best_score 判定"有依据"时，0.50 分的噪声块照样会进 prompt 被照抄；
    #: 0 表示不过滤（始终保留最高分那一块）。
    context_min_score: float = 0.45
    #: 词面覆盖闸门：问题里的**内容词**至少要有这个比例出现在召回片段里，
    #: 否则判定"与知识库无关"直接拒答（解决"问天气却答休学"）。
    #: 0 表示关闭。默认 0.30 偏保守——宁可多拒一次，也不硬答无关问题。
    min_lexical_coverage: float = 0.30
    #: 精排分数高到这个程度时，跳过词面覆盖闸门（0 = 永不跳过）。
    #: 加这条是因为 48 篇合成语料 + 73 题实测：口语化提问（"年前攒的一沓票子
    #: 一直忘了交"）的候选 Hit@1 有 92.6%，但内容词一个都不在原文里，
    #: 于是 85% 的**可答题**被覆盖闸门判成 off_topic。词面重叠是"证据是否相关"的
    #: 廉价代理，交叉编码器是直接测量它——后者已经给出高置信时再要求字面重合，
    #: 等于让代理否决正主。
    #: 取 0.45（= score_threshold）：要让"精排已认"早于阈值生效。只把这条线降到 0.45
    #: 而不动阈值时无用——0.45–0.55 那一段先被 below_threshold 拒掉，轮不到放宽
    #: （reports/ablation-relax045.md：仍是 83.6%）。两条一起改才是 86.6%。
    #: 精排关掉时（gate=vector_cosine）这条不生效，词面闸门仍是唯一的跑题代理。
    coverage_relax_score: float = 0.45
    #: 同义词表：把用户口语换回制度用词（"进校"→"入校"、"来访"→"访客"）。
    #: 起因：问"我想进校访问"命中的却是"校外访问电子数据库需 VPN"——分词后
    #: 「进校」和「入校」是两个词，BM25 不给分、覆盖闸门也判没命中。
    synonyms_path: str = "config/synonyms.yaml"
    userdict_path: str = "config/jieba_userdict.txt"
    stopwords_path: str = "config/stopwords_zh.txt"


@dataclass(frozen=True)
class ChunkingConfig:
    """结构化切块参数。"""

    max_tokens: int = 700
    min_tokens: int = 120
    overlap_tokens: int = 60
    keep_tables_whole: bool = True
    table_max_tokens: int = 1200
    #: 单个表格 chunk 最多包含多少行数据（超出按行分组并重复表头）。
    #: 整表一块会让"某一行"被稀释，检索时容易命中同表的其他行；
    #: 实测 4 行一组比整表更适合"限额/审批人"这类行级提问。0 = 不按行数限制。
    table_max_rows: int = 4


@dataclass(frozen=True)
class SessionConfig:
    """会话存储参数。"""

    store: str = "memory"
    sqlite_path: str = "data/index/sessions.db"
    max_turns: int = 6
    summarize_after: int = 6


@dataclass(frozen=True)
class GenerationConfig:
    """生成与引用溯源的业务规则。

    这些文案由**代码**追加/替换，不交给 LLM 自由发挥：
    拒答话术必须逐字一致（评测要按字符串匹配），免责声明必须 100% 出现。
    """

    max_context_chunks: int = 5
    max_context_tokens: int = 3000
    require_citation: bool = True
    quote_chars: int = 120
    add_disclaimer: bool = True
    # ---- claim 级引用校验（闸门 2.5，阶段 G）----
    # 闸门 2 只查"有没有合法 [n]"，拦不住"编号对、内容错位"。逐句比对结论与
    # 所引章节原文，不达标先尝试改标，仍不行按模式处理。
    claim_check: str = "fix"            # off | warn | fix | strict
    claim_min_coverage: float = 0.6     # 句子内容词出现在引用章节里的比例下限
    claim_min_tokens: int = 3           # 少于这么多内容词的句子不判（套话/过渡句）
    claim_reassign_margin: float = 0.15 # 改标需比原引用高出的覆盖差
    claim_max_drop_ratio: float = 0.5   # strict 模式删句比例上限，超出则退回只改标
    claim_number_guard: bool = True     # 金额/天数等阿拉伯数字必须出现在原文
    refuse_text: str = "当前知识库中未找到相关依据，建议咨询对应部门或人工客服。"
    disclaimer: str = "提示：以上内容依据知识库原文整理，具体以原文件为准，请以最新发布版本为准。"


@dataclass(frozen=True)
class SyncConfig:
    """知识库同步参数。"""

    mode: str = "schedule"
    interval_seconds: int = 60
    watch_dir: str = "data/raw"
    enable_watcher: bool = False
    watcher_debounce_s: float = 2.0
    include_patterns: tuple[str, ...] = ("*.pdf", "*.docx", "*.md", "*.txt", "*.xlsx")
    exclude_dirs: tuple[str, ...] = ("__pycache__", ".git", "~$")
    default_doc_type: str = "policy"
    # ---- 并发控制 ----
    lock_file: str = "data/index/.sync.lock"
    write_lock_timeout_s: float = 30.0
    read_lock_timeout_s: float = 20.0
    stale_lock_s: float = 900.0
    # ---- 提交批大小：每处理这么多文件就短暂持写锁提交一次 ----
    commit_batch_docs: int = 20
    # ---- 同一份文件的多个年份/版本：只索引最新的一份 ----
    prefer_latest_version: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """日志参数。"""

    level: str = "INFO"
    dir: str = "logs"
    retain_days: int = 14
    qa_log: str = "qa.jsonl"


@dataclass(frozen=True)
class ServerConfig:
    """HTTP 服务参数。"""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("*",)
    #: 非空时，写操作与问答接口需要携带该令牌（公网暴露的最低保护）
    public_access_token: str = field(default="", repr=False)
    #: 每 IP 每分钟请求上限，0 表示不限
    rate_limit_per_min: int = 60
    #: 是否信任 X-Forwarded-For。只有确实部署在反向代理后面（且代理会重写该头）
    #: 才设 true；否则任何人都能伪造它绕过登录节流与限流。
    trust_forwarded_for: bool = False
    #: 交给 uvicorn ``--forwarded-allow-ips``：只有来自这些地址的请求，其
    #: ``X-Forwarded-For`` / ``X-Forwarded-Proto`` 才会被用来改写连接地址与协议。
    #: 默认只信任本机代理（cloudflared / nginx / caddy 常见于同机部署）。
    #: 填 "*" 等于对所有来源放行，限流与登录节流随即失效，不要这么配。
    forwarded_allow_ips: tuple[str, ...] = ("127.0.0.1",)
    #: 对外域名（如 https://zhidu.example.edu.cn）。仅用于在控制台/健康检查里
    #: 展示"分享给别人的地址"，不参与路由；反向代理后本机看不到真实域名，
    #: 这里显式配置比自动探测可靠。
    public_url: str = ""


@dataclass(frozen=True)
class BrandingConfig:
    """品牌信息（页面标题、配色、校名校训）。

    为什么做成配置而不是写死在 HTML 里：这套系统会部署到不同学校/学院，
    换校名不该改前端代码；同时页面顶部也要展示"数据与联系方式以官方为准"。
    """

    school_name: str = "郑州大学"
    short_name: str = "郑大"
    motto: str = "求是 担当"
    site_title: str = "制度与办事流程问答"
    primary_color: str = "#0d4c92"
    accent_color: str = "#c8a45c"
    #: 页面底部的免责说明（不参与 LLM 生成，纯展示）
    footer_note: str = "回答依据知识库原文整理，具体以学校官方文件为准"


@dataclass(frozen=True)
class DepartmentsConfig:
    """部门目录（关键词 → 部门 → 联系方式）。

    命中只做"提示"，不进入回答正文：把联系方式塞进 answer 会污染忠实度评测，
    也会让模型的话术变得不可控。结构化返回，由前端渲染成卡片。
    """

    enabled: bool = True
    config_file: str = "config/departments.yaml"
    #: 一次最多提示几个部门（问题里同时出现"教务处""财务处"是常见的）
    max_suggestions: int = 3


@dataclass(frozen=True)
class DeptConsoleConfig:
    """部门看板（/dept）：近 N 天的问题分布与周报预览。"""

    enabled: bool = True
    #: 非空时 /dept/* 数据接口需要 X-Access-Token 或 ?token=；空表示不校验（内网演示）
    token: str = field(default="", repr=False)
    window_days: int = 30
    top_n: int = 12
    #: 拖拽上传：单个文件大小上限（MB），必须 >=1
    upload_max_mb: int = 20
    #: 拖拽上传：每个账号每分钟最多上传多少个文件，0 表示不限
    upload_rate_per_min: int = 20


@dataclass(frozen=True)
class ReportConfig:
    """周报投递：每周把「本部门被问什么」发给部门联系人。

    默认 **不启用也不真发**：`enabled=false` 不排期，`dry_run=true` 只把邮件
    写进 `outbox_dir`（.eml 可用邮件客户端打开）。要真发必须同时配好 SMTP。
    """

    enabled: bool = False
    weekday: str = "mon"
    hour: int = 9
    minute: int = 0
    days: int = 7
    dry_run: bool = True
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_ssl: bool = True
    smtp_user: str = ""
    #: 密码从环境变量读，不写进 YAML
    smtp_password_env: str = "SMTP_PASSWORD"
    from_addr: str = ""
    subject_prefix: str = "【学生咨询周报】"
    outbox_dir: str = "logs/outbox"

@dataclass(frozen=True)
class AuthConfig:
    """登录与角色权限。

    三种角色：student（只用问答）/ dept·admin（问答 + 部门看板 + 运维台）/ guest（只问答）。
    ``enabled=False`` 时不做登录校验，按本机管理员处理（仅开发/自动化测试用）。
    """

    enabled: bool = True
    users_file: str = "config/users.yaml"
    session_ttl_hours: float = 12.0
    allow_guest: bool = True
    cookie_name: str = "rag_session"
    #: 强制给会话 Cookie 打 Secure 位。留 false 也可以：请求本身走 HTTPS 时
    #: 代码会自动加 Secure，明文 HTTP 局域网演示则不会（否则永远登不上）。
    cookie_secure: bool = False
    login_limit_per_min: int = 10

# --------------------------------------------------------------------------
# 顶层 Settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    """全局配置对象。通过 :func:`get_settings` 获取单例。"""

    root: Path = PROJECT_ROOT
    project: ProjectConfig = field(default_factory=ProjectConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    branding: BrandingConfig = field(default_factory=BrandingConfig)
    departments: DepartmentsConfig = field(default_factory=DepartmentsConfig)
    dept_console: DeptConsoleConfig = field(default_factory=DeptConsoleConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # ---- 路径解析 ----
    def resolve(self, value: str | Path) -> Path:
        """把配置里的相对路径解析为绝对路径（相对项目根目录）。"""
        path = Path(value)
        return path if path.is_absolute() else (self.root / path)

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.sync.watch_dir)

    @property
    def index_dir(self) -> Path:
        return self.resolve(self.vectorstore.persist_dir).parent

    @property
    def chroma_dir(self) -> Path:
        return self.resolve(self.vectorstore.persist_dir)

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def log_dir(self) -> Path:
        return self.resolve(self.logging.dir)

    @property
    def users_file(self) -> Path:
        """登录用户库（只有密码哈希）。"""
        return self.resolve(self.auth.users_file)

    @property
    def departments_file(self) -> Path:
        """部门目录 YAML 路径（缺失时功能降级：不提示联系方式）。"""
        return self.resolve(self.departments.config_file)

    @property
    def outbox_dir(self) -> Path:
        """周报未真发时的落盘目录（.eml + .md）。"""
        return self.resolve(self.report.outbox_dir)

    @property
    def reports_dir(self) -> Path:
        """评测报告目录（`reports/eval-*.md`），运维台从中取最近一次离线结果。"""
        return self.resolve("reports")

    @property
    def qa_log_path(self) -> Path:
        return self.log_dir / self.logging.qa_log

    # ---- 派生信息 ----
    @property
    def embedding_fingerprint(self) -> str:
        """embedding 指纹，用于隔离不同模型的向量集合（方案假设 #17）。

        必须把 provider 也算进去：mock（哈希向量）与真实 bge-m3 维度相同但语义
        完全不同，如果共用一个 collection，mock 模式下建的索引会污染真实索引。
        """
        provider = self.embedding.provider.strip().lower()
        if provider in {"mock", "hash"}:
            return f"mock-{self.embedding.dim}"
        name = self.embedding.model.replace("/", "-").replace("BAAI-", "")
        return f"{name}-{self.embedding.dim}"

    @property
    def collection_name(self) -> str:
        """真实 collection 名 = 配置名 + embedding 指纹。"""
        return f"{self.vectorstore.collection}__{self.embedding_fingerprint}"

    def validate(self) -> None:
        """校验跨字段约束，配置写错时尽早失败。"""
        c, r, s = self.chunking, self.retrieval, self.sync
        if c.max_tokens <= 0:
            raise ConfigError(f"chunking.max_tokens 必须为正数，当前 {c.max_tokens}")
        if not 0 <= c.overlap_tokens < c.max_tokens:
            raise ConfigError(
                f"chunking.overlap_tokens({c.overlap_tokens}) 必须 >=0 且小于 max_tokens({c.max_tokens})"
            )
        if c.min_tokens < 0 or c.min_tokens > c.max_tokens:
            raise ConfigError("chunking.min_tokens 必须在 [0, max_tokens] 区间")
        if c.table_max_tokens < c.max_tokens:
            raise ConfigError("chunking.table_max_tokens 不应小于 max_tokens")
        if c.table_max_rows < 0:
            raise ConfigError("chunking.table_max_rows 不能为负（0 = 不按行数切分）")
        if r.top_n <= 0 or r.candidate_k <= 0:
            raise ConfigError("retrieval.top_n / candidate_k 必须为正整数")
        if r.rrf_k <= 0:
            raise ConfigError("retrieval.rrf_k 必须为正整数")
        if r.weight_vector < 0 or r.weight_bm25 < 0:
            raise ConfigError("检索融合权重不能为负")
        if r.weight_vector == 0 and r.weight_bm25 == 0:
            raise ConfigError("检索融合权重不能同时为 0")
        if not 0.0 <= r.score_threshold <= 1.0:
            raise ConfigError("retrieval.score_threshold 必须落在 [0, 1]")
        if not 0.0 <= r.context_min_score <= 1.0:
            raise ConfigError("retrieval.context_min_score 必须落在 [0, 1]（0 = 不过滤）")
        if not 0.0 <= r.min_lexical_coverage <= 1.0:
            raise ConfigError("retrieval.min_lexical_coverage 必须落在 [0, 1]（0 = 关闭该闸门）")
        if not 0.0 <= self.retrieval.coverage_relax_score <= 1.0:
            raise ConfigError("retrieval.coverage_relax_score 必须落在 [0, 1]（0 = 精排高分时不再跳过覆盖闸门）")
        if self.embedding.dim <= 0:
            raise ConfigError("embedding.dim 必须为正整数")
        if self.reranker.batch_size < 1:
            raise ConfigError("reranker.batch_size 至少为 1")
        if self.reranker.max_length < 32:
            raise ConfigError("reranker.max_length 至少为 32")
        if s.interval_seconds < 5:
            raise ConfigError("sync.interval_seconds 建议 >=5 秒，避免空转")
        if s.mode not in {"schedule", "watch", "off"}:
            raise ConfigError(f"sync.mode 仅支持 schedule/watch/off，当前 {s.mode}")
        if s.write_lock_timeout_s <= 0 or s.read_lock_timeout_s <= 0:
            raise ConfigError("锁等待超时必须为正数")
        if s.commit_batch_docs < 1:
            raise ConfigError("sync.commit_batch_docs 至少为 1")
        if self.logging.level.upper() not in _LOG_LEVELS:
            raise ConfigError(f"logging.level 非法：{self.logging.level}")
        if self.session.store not in {"memory", "sqlite"}:
            raise ConfigError(f"session.store 仅支持 memory/sqlite，当前 {self.session.store}")
        if self.generation.max_context_chunks <= 0:
            raise ConfigError("generation.max_context_chunks 必须为正整数")
        if not self.generation.refuse_text.strip():
            raise ConfigError("generation.refuse_text 不能为空（拒答话术参与评测匹配）")
        if self.session.max_turns < 1:
            raise ConfigError("session.max_turns 至少为 1")
        if self.dept_console.upload_max_mb < 1:
            raise ConfigError("dept_console.upload_max_mb 至少为 1（MB）")
        if self.dept_console.upload_rate_per_min < 0:
            raise ConfigError("dept_console.upload_rate_per_min 不能为负（0 = 不限）")


# --------------------------------------------------------------------------
# YAML / 环境变量加载与类型转换
# --------------------------------------------------------------------------
def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件，返回 dict。空文件视为 {}。"""
    if not path.exists():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - 语法错误分支
        raise ConfigError(f"YAML 解析失败：{path} -> {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射（dict）：{path}")
    return data


def _deep_set(target: dict[str, Any], dotted: str, value: Any) -> None:
    """按 ``a.b.c`` 路径写入嵌套 dict，缺失的中间层自动创建。"""
    parts = dotted.split(".")
    cursor = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def _parse_scalar(value: str) -> Any:
    """把环境变量字符串转成合适的标量（尽量兼容 bool/int/float/json）。"""
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~", ""}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def collect_env_overrides(env: Mapping[str, str]) -> dict[str, str]:
    """收集环境变量覆盖项。

    - ``DEEPSEEK_API_KEY``     -> ``llm.api_key``
    - ``DEEPSEEK_BASE_URL``    -> ``llm.base_url``
    - ``RAG_<SECTION>__<KEY>`` -> ``<section>.<key>``（大小写不敏感）
    - ``RAG_FORCE_MOCK=1``     -> ``llm.provider=mock``（调试用快捷开关）
    """
    overrides: dict[str, str] = {}
    if env.get("DEEPSEEK_API_KEY"):
        overrides["llm.api_key"] = env["DEEPSEEK_API_KEY"].strip()
    if env.get("DEEPSEEK_BASE_URL"):
        overrides["llm.base_url"] = env["DEEPSEEK_BASE_URL"].strip()

    force_mock = (env.get("RAG_FORCE_MOCK") or "").strip().lower()
    if force_mock in {"1", "true", "yes", "on"}:
        overrides["llm.provider"] = "mock"
        overrides["embedding.provider"] = "mock"
        overrides["reranker.provider"] = "mock"

    for key, value in env.items():
        if not key.upper().startswith("RAG_") or "__" not in key:
            continue
        section, _, rest = key[4:].partition("__")
        if not section or not rest:
            continue
        overrides[f"{section.lower()}.{rest.lower().replace('__', '.')}"] = value
    return overrides


def _unwrap_optional(annotation: Any) -> Any:
    """把 ``Optional[T]`` / ``T | None`` 解成 ``T``。"""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _coerce(value: Any, annotation: Any) -> Any:
    """按 dataclass 字段注解做宽松类型转换。

    这样 ``RAG_RETRIEVAL__TOP_N=8``（字符串）也能正确变成 int 8。
    """
    annotation = _unwrap_optional(annotation)
    if value is None:
        return None

    origin = get_origin(annotation)
    if origin in (tuple, list):
        inner = _unwrap_optional(get_args(annotation)[0]) if get_args(annotation) else str
        if isinstance(value, str):
            items: list[Any] = [p.strip() for p in value.split(",") if p.strip()]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = [value]
        return tuple(_coerce(i, inner) for i in items) if origin is tuple else [
            _coerce(i, inner) for i in items
        ]

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        if isinstance(value, dict):
            return _build_dataclass(annotation, value)
        raise ConfigError(f"配置项应为映射，实际为：{type(value).__name__}")

    if annotation is bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if annotation is int:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"期望整数，实际为 {value!r}") from exc
    if annotation is float:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"期望浮点数，实际为 {value!r}") from exc
    if annotation is str:
        if isinstance(value, (list, tuple)):
            raise ConfigError(f"期望字符串，实际为列表：{value!r}")
        return str(value).strip()
    if annotation is Path:
        return Path(str(value))
    return value


def _build_dataclass(cls: type[T], data: Mapping[str, Any]) -> T:
    """依据 dataclass 字段名从 dict 构造实例；未知键直接报错（防拼写错误）。"""
    hints = get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ConfigError(f"{cls.__name__} 存在未知配置项：{sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name in field_names:
        if name in data:
            kwargs[name] = _coerce(data[name], hints.get(name, Any))
    return cls(**kwargs)  # type: ignore[return-value]


def build_settings(
    data: Mapping[str, Any],
    *,
    root: Path = PROJECT_ROOT,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """从原始 dict + 环境变量构造并校验 :class:`Settings`。

    单独暴露此函数便于测试注入，不需要落盘配置文件。
    """
    merged: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    env_map = dict(os.environ if env is None else env)
    for dotted, raw in collect_env_overrides(env_map).items():
        _deep_set(merged, dotted, _parse_scalar(raw))

    # api_key 单独从环境变量注入（配置里只存变量名）
    llm_raw = merged.get("llm")
    if isinstance(llm_raw, dict):
        key_env = str(llm_raw.get("api_key_env", "DEEPSEEK_API_KEY"))
        if not llm_raw.get("api_key"):
            llm_raw["api_key"] = env_map.get(key_env, "").strip()

    settings = _build_dataclass(Settings, {**merged, "root": root})
    settings.validate()
    return settings


def load_settings(
    config_path: Path | None = None,
    *,
    root: Path | None = None,
    env_file: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """加载完整配置：先读 ``.env``，再读 YAML，再套用环境变量覆盖。"""
    resolved_root = Path(root) if root is not None else PROJECT_ROOT
    dotenv_path = env_file if env_file is not None else resolved_root / ".env"
    if dotenv_path.exists():
        # override=False：已存在的真实环境变量优先，.env 只做兜底
        load_dotenv(dotenv_path=dotenv_path, override=False)
    path = Path(config_path) if config_path is not None else resolved_root / "config" / "settings.yaml"
    data = load_yaml(path)
    return build_settings(data, root=resolved_root, env=os.environ if env is None else env)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取进程内配置单例。测试中如需重载可调用 ``get_settings.cache_clear()``。"""
    return load_settings()
