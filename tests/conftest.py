"""pytest 公共 fixture。

约定：
- 所有测试在 **mock 模式**下运行（不联网、不下载模型、不需要 API Key）；
- 每个用例用独立 ``tmp_path`` 作为项目根，避免污染真实的 ``data/`` 与 ``logs/``；
- 测试默认关闭定时同步与目录监听（``sync.mode=off``），需要验证自动化的用例
  自己在 settings 上打开，避免后台线程干扰断言。

（注：本文件曾因并行编辑出现过重复的 ``mock_settings`` 定义与一个引用已废弃 API
的 ``pipeline`` fixture，已合并清理；如再出现重复定义，后定义会静默覆盖前者，
请勿再加第二个。）
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

# 必须在导入任何 rag_assistant 模块之前设置：模块导入期（例如 api.main 里
# 模块级的 create_app）就可能调用 setup_logging，那时日志目录已经定了。
# 指向临时目录后，跑测试永远不会再往真实的 logs/ 里写东西。
_TEST_LOG_DIR = Path(tempfile.mkdtemp(prefix="rag-test-logs-"))
os.environ.setdefault("RAG_LOG_DIR", str(_TEST_LOG_DIR))

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for extra in (SRC_DIR, SCRIPTS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rag_assistant.config import PROJECT_ROOT as CONFIG_ROOT  # noqa: E402
from rag_assistant.config import Settings, build_settings, load_yaml  # noqa: E402
from rag_assistant import logging_setup as _logging_setup  # noqa: E402


@pytest.fixture(autouse=True)
def reset_logging_between_tests() -> Iterator[None]:
    """每个用例前后复位 logging 的"已配置"标志。

    ``setup_logging`` 有全局幂等标志：只要有一个用例用真实项目根配置过日志，
    后面的用例即使指向 tmp 项目根也不会重新配置，日志会写进真实的 ``logs/``
    （运维台的告警面板里就混进了 pytest 的临时路径）。逐用例复位即可。
    """
    _logging_setup._CONFIGURED = False
    yield
    _logging_setup._CONFIGURED = False


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """仓库根目录（只读使用）。"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def base_config() -> dict[str, Any]:
    """真实 settings.yaml 的内容（每次测试自行覆盖需要的字段）。"""
    return load_yaml(CONFIG_ROOT / "config" / "settings.yaml")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """临时项目根：含 config/settings.yaml、logging.yaml 与空的 data/raw。"""
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_ROOT / "config" / "settings.yaml", tmp_path / "config" / "settings.yaml")
    shutil.copy(CONFIG_ROOT / "config" / "logging.yaml", tmp_path / "config" / "logging.yaml")
    # 部门目录（阶段 H）：缺失时功能会静默降级，但测试需要它存在才能验证联系方式
    departments = CONFIG_ROOT / "config" / "departments.yaml"
    if departments.exists():
        shutil.copy(departments, tmp_path / "config" / "departments.yaml")
    users = CONFIG_ROOT / "config" / "users.yaml"
    if users.exists():
        shutil.copy(users, tmp_path / "config" / "users.yaml")
    return tmp_path


@pytest.fixture()
def settings(project: Path, base_config: dict[str, Any]) -> Settings:
    """指向临时目录的 Settings（索引/日志都落在 tmp 下；默认走配置里的真实组件）。"""
    data = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base_config.items()}
    data["vectorstore"] = {**data["vectorstore"], "persist_dir": "data/index/chroma"}
    data["logging"] = {**data["logging"], "dir": "logs"}
    data["sync"] = {**data["sync"], "watch_dir": "data/raw"}
    data["session"] = {**data["session"], "sqlite_path": "data/index/sessions.db"}
    return build_settings(data, root=project, env={})


@pytest.fixture()
def mock_settings(project: Path, base_config: dict[str, Any]) -> Settings:
    """全 mock 链路：哈希向量 + 内存向量库 + 词面重排 + MockLLM。

    向量维度降到 256、候选数降到 12 只为让测试快一点，逻辑与生产一致。
    """
    data = {key: (dict(value) if isinstance(value, dict) else value) for key, value in base_config.items()}
    data["llm"] = {**data["llm"], "provider": "mock"}
    data["embedding"] = {**data["embedding"], "provider": "mock", "model": "mock", "dim": 256}
    data["reranker"] = {**data["reranker"], "provider": "mock", "enabled": True, "top_k_in": 15}
    # 向量库用**真实 Chroma**（落在 tmp 目录）：跨实例持久化是生产行为，
    # 而 memory 后端重启即空，会让"清单/向量/BM25 三方一致"这类校验失去意义。
    data["vectorstore"] = {**data["vectorstore"], "persist_dir": "data/index/chroma"}
    data["retrieval"] = {
        **data["retrieval"],
        "candidate_k": 12,
        # 启发式 reranker 的分数尺度远低于 bge-reranker 的 sigmoid 输出，
        # 阈值必须各自标定：0.20 是本测试集下"能召回正确条款、又能否掉无关问"的取值。
        # 真实 bge-reranker 的默认阈值仍是 0.50（config 默认值）。
        "score_threshold": 0.20,
        # 生产配置用 context_min_score 把低分块挡在 prompt 外；测试关注的是组装逻辑本身，
        # 这里关掉（0 = 不过滤），过滤行为由 tests/test_context.py 单独覆盖。
        "context_min_score": 0.0,
    }
    data["logging"] = {**data["logging"], "dir": "logs"}
    data["sync"] = {**data["sync"], "watch_dir": "data/raw", "mode": "off", "enable_watcher": False}
    # 鉴权默认关闭：老用例关注的是业务行为；登录/角色有专门的 test_auth.py
    data["auth"] = {**data.get("auth", {}), "enabled": False}
    return build_settings(data, root=project, env={})


@pytest.fixture()
def raw_dir(project: Path) -> Path:
    """空的知识库目录。"""
    return project / "data" / "raw"


@pytest.fixture()
def sample_raw_dir(repo_root: Path, raw_dir: Path) -> Path:
    """把仓库示例文档复制到临时 raw 目录，并现场生成 DOCX / XLSX。

    现场生成是为了让测试不依赖仓库里的二进制文件（换机器/CI 也能跑）。
    """
    source = repo_root / "data" / "raw"
    for item in source.iterdir():
        if item.is_file() and item.suffix.lower() in {".md", ".txt", ".pdf"}:
            shutil.copy(item, raw_dir / item.name)

    import gen_sample_docs

    gen_sample_docs.build_docx(raw_dir / "科研经费报销管理办法.docx")
    gen_sample_docs.build_xlsx(raw_dir / "差旅与会议费标准表.xlsx")
    return raw_dir


def write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """测试辅助：写 YAML。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# 最小可用 PDF（纯 ASCII，无第三方依赖）
# --------------------------------------------------------------------------
def build_minimal_pdf(lines: list[str]) -> bytes:
    """生成一个单页、可用 pypdf 正确提取文本的 PDF。

    只用于测试 PDF 解析器；中文需要嵌入字体，故这里用英文内容验证
    "文本提取 + 页码 + 章节兜底"这条链路本身。
    """
    content_parts = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    for line in lines:
        safe = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_parts.append(f"({safe}) Tj T*")
    content_parts.append("ET")
    stream = "\n".join(content_parts).encode("latin-1")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    )
    return bytes(out)


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """最小 PDF 文件。"""
    path = tmp_path / "policy_en.pdf"
    path.write_bytes(
        build_minimal_pdf(
            [
                "Chapter 1 General Provisions",
                "Article 1 This regulation applies to all enrolled students.",
                "Article 2 The registrar office is responsible for records.",
                "Chapter 2 Registration",
                "Article 3 Students must register within two weeks.",
            ]
        )
    )
    return path
