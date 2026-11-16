"""阶段 C 演示：多轮对话 + 引用溯源 + 拒答 + 流式。

用法::

    python scripts/chat_demo.py                       # 默认阈值 0.15（mock 链路）
    python scripts/chat_demo.py --threshold 0.5       # 用配置里的真实模型阈值体验拒答
    python scripts/chat_demo.py --stream              # 看流式事件
    python scripts/chat_demo.py --session demo-1      # 指定会话 id

说明：mock 链路的分数与真实模型不可比（见 README 6.3），所以演示默认把阈值降到 0.15，
让"能回答的问题"真的走完生成流程；用 0.5 则几乎全部判为拒答——那不代表检索错。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_assistant.chat_service import ChatService  # noqa: E402
from rag_assistant.config import get_settings  # noqa: E402
from rag_assistant.kb import KnowledgeBase  # noqa: E402
from rag_assistant.logging_setup import setup_logging  # noqa: E402

DEMO_QUESTIONS = [
    "休学最长可以多久？",
    "那复学呢？需要什么材料？",
    "在读证明咋开，多久能拿到？",
    "差旅住宿费报销标准是多少？",
    "图书馆几点开门？",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="多轮问答演示")
    parser.add_argument("--session", default="demo", help="会话 ID（不同 ID 之间上下文隔离）")
    parser.add_argument("--threshold", type=float, default=0.12, help="相关性阈值（mock 链路建议 0.12）")
    parser.add_argument("--stream", action="store_true", help="用流式接口演示")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    kb = KnowledgeBase(settings)
    if kb.status()["vector_chunks"] == 0:
        print("索引为空，先执行一次同步……")
        kb.sync()
    service = ChatService(settings, kb=kb)

    try:
        print(f"\n会话：{args.session}   阈值：{args.threshold}   LLM：{service.llm.model}")
        print("注意：mock 链路的分数口径与真实 bge 模型不可比，这里默认 0.12 只为让演示可读；")
        print("      生产阈值必须用真实模型 + 评测集标定（阶段 F）。")
        print("=" * 72)
        for question in DEMO_QUESTIONS:
            if args.stream:
                _stream_once(service, args.session, question, args.threshold)
            else:
                _chat_once(service, args.session, question, args.threshold)
        print("\n=== 会话历史（已记录轮次）===")
        for index, turn in enumerate(service.sessions.list_turns(args.session), start=1):
            flag = "（拒答）" if turn.refused else ""
            print(f"{index}. {turn.question}{flag}  ->  改写：{turn.rewritten_query}")
    finally:
        service.close()
    return 0


def _chat_once(service: ChatService, session: str, question: str, threshold: float) -> None:
    reply = service.chat(session, question, threshold=threshold)
    print(f"\n问：{question}")
    if reply.rewritten_query != question:
        print(f"  ↳ 改写后：{reply.rewritten_query}")
    if reply.extra_queries:
        print(f"  ↳ 等价查询：{reply.extra_queries}")
    print(f"答：{reply.answer}")
    print(
        f"〔{'拒答' if reply.refused else '已作答'}｜引用 {len(reply.citations)} 条｜"
        f"{reply.latency_ms:.0f}ms｜tokens {reply.usage_prompt}+{reply.usage_completion}〕"
    )


def _stream_once(service: ChatService, session: str, question: str, threshold: float) -> None:
    print(f"\n问：{question}")
    print("答（流式）：", end="", flush=True)
    for event in service.chat_stream(session, question, threshold=threshold):
        kind = event.get("type")
        if kind == "meta":
            print(f"\n  ↳ 改写后：{event.get('rewritten_query')}")
        elif kind == "token":
            print(event.get("text", ""), end="", flush=True)
        elif kind == "retract":
            print(f"\n  [前端需替换为拒答] {event.get('answer')}")
        elif kind == "done":
            print(f"\n〔{'拒答' if event.get('refused') else '已作答'}｜"
                  f"引用 {len(event.get('citations') or [])} 条｜{event.get('latency_ms')}ms〕")


if __name__ == "__main__":
    raise SystemExit(main())
