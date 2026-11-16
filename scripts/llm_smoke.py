"""探测任意 OpenAI 兼容端点（LM Studio / Ollama / DeepSeek）是否可用、够不够"听话"。

为什么需要它：换模型最容易踩的坑不是"连不上"，而是
1. **思考型模型把内容写进 reasoning_content，content 为空** → 本系统拿到空答案；
2. **格式不遵守**：改写不输出 JSON、回答不带 `[n]` 编号、无依据时不拒答。

这个脚本用真实 prompt 当场验这四条，并给出延迟与 tokens/s。

用法::

    python scripts\\llm_smoke.py                       # 默认本地 LM Studio 1234
    python scripts\\llm_smoke.py --model <id>
    python scripts\\llm_smoke.py --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b
    python scripts\\llm_smoke.py --base-url https://api.deepseek.com --model deepseek-chat --api-key sk-xxx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import requests  # noqa: E402

from rag_assistant.llm.prompts import answer_prompt, rewrite_prompt  # noqa: E402

REFUSE = "当前知识库中未找到相关依据，建议咨询对应部门或人工客服。"
CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def _headers(api_key: str | None) -> dict[str, str]:
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}


def list_models(base_url: str, api_key: str | None) -> list[str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/models", headers=_headers(api_key), timeout=10)
        response.raise_for_status()
        return [item.get("id", "") for item in response.json().get("data", [])]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 无法列出模型：{type(exc).__name__}: {exc}")
        return []


def chat(base_url: str, model: str, messages: list[dict], api_key: str | None, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=_headers(api_key),
        data=json.dumps(payload),
        timeout=300,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    choice = body["choices"][0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    return {
        "text": (message.get("content") or "").strip(),
        "reasoning": (message.get("reasoning_content") or message.get("reasoning") or "").strip(),
        "elapsed": elapsed,
        "usage": usage,
        "finish": choice.get("finish_reason"),
    }


def _report(label: str, result: dict) -> None:
    completion = result["usage"].get("completion_tokens") or 0
    speed = completion / result["elapsed"] if result["elapsed"] else 0
    print(
        f"    耗时 {result['elapsed']:.1f}s｜completion {completion} tok｜{speed:.1f} tok/s"
        f"｜finish={result['finish']}｜思考 {len(result['reasoning'])} 字"
    )
    if not result["text"] and result["reasoning"]:
        print("    ⚠ content 为空但 reasoning 非空 → 思考型模型未产出答案，本系统会判拒答")
    print(f"    输出：{result['text'][:200] if result['text'] else '（空）'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI 兼容端点冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None, help="缺省用 /models 里第一个非 embedding 模型")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=1500, help="思考型模型需要更大预算")
    args = parser.parse_args(argv)

    print(f"端点：{args.base_url}")
    models = list_models(args.base_url, args.api_key)
    if models:
        print(f"可用模型：{', '.join(models)}")
    model = args.model or next((m for m in models if "embed" not in m.lower()), None)
    if not model:
        print("[fail] 没有可用模型：先在 LM Studio 里加载模型并 Start Server")
        return 2
    print(f"使用模型：{model}\n")

    results: list[tuple[str, bool, str]] = []

    # ---- 1. 连通 + 是否产出 content ----
    print("[1] 基础对话（检查是否产出 content）")
    try:
        first = chat(args.base_url, model, [{"role": "user", "content": "用一句中文回答：学院教务处负责什么？"}], args.api_key, args.max_tokens)
        _report("基础", first)
        ok = bool(first["text"])
        results.append(("产出 content（非纯思考）", ok, f"{first['elapsed']:.1f}s"))
    except Exception as exc:  # noqa: BLE001
        print(f"    失败：{type(exc).__name__}: {exc}")
        return 1

    # ---- 2. 改写必须输出 JSON ----
    print("\n[2] 问题改写（必须输出 JSON）")
    messages = [
        m.to_dict()
        for m in rewrite_prompt("那它的期限呢？", "用户：休学最长可以多久\n助手：休学期一般以一年为期。", 3)
    ]
    try:
        result = chat(args.base_url, model, messages, args.api_key, args.max_tokens)
        _report("改写", result)
        parsed = None
        start, end = result["text"].find("{"), result["text"].rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(result["text"][start : end + 1])
            except json.JSONDecodeError:
                parsed = None
        ok = isinstance(parsed, dict) and bool(parsed.get("standalone"))
        results.append(("改写输出 JSON", ok, f"{result['elapsed']:.1f}s"))
    except Exception as exc:  # noqa: BLE001
        print(f"    失败：{type(exc).__name__}: {exc}")
        results.append(("改写输出 JSON", False, "调用失败"))

    # ---- 3. 回答必须带 [n] 引用 ----
    print("\n[3] 带资料回答（必须带 [n] 引用）")
    blocks = [
        "[1] 文档：学生学籍管理规定.md ｜ 章节：第三章 休学、复学与退学 > 第七条 休学条件 ｜ 页码：无\n"
        "正文：学生因病需停课治疗的应当办理休学。休学期一般以一年为期，累计不得超过两年。"
    ]
    messages = [m.to_dict() for m in answer_prompt("休学最长可以多久？", blocks, refuse_text=REFUSE)]
    try:
        result = chat(args.base_url, model, messages, args.api_key, args.max_tokens)
        _report("回答", result)
        cited = CITATION_RE.findall(result["text"])
        results.append(("回答带 [n] 引用", bool(cited), f"{result['elapsed']:.1f}s"))
        print(f"    引用编号：{cited or '无'}")
    except Exception as exc:  # noqa: BLE001
        print(f"    失败：{type(exc).__name__}: {exc}")
        results.append(("回答带 [n] 引用", False, "调用失败"))

    # ---- 4. 无依据必须拒答 ----
    print("\n[4] 无依据问题（必须拒答而不是编造）")
    messages = [m.to_dict() for m in answer_prompt("图书馆几点开门？", ["（无相关资料）"], refuse_text=REFUSE)]
    try:
        result = chat(args.base_url, model, messages, args.api_key, args.max_tokens)
        _report("拒答", result)
        text = result["text"]
        ok = REFUSE[:12] in text or "未找到" in text or "没有" in text or "无法" in text
        results.append(("无依据时拒答", ok, f"{result['elapsed']:.1f}s"))
    except Exception as exc:  # noqa: BLE001
        print(f"    失败：{type(exc).__name__}: {exc}")
        results.append(("无依据时拒答", False, "调用失败"))

    # ---- 汇总 ----
    passed = sum(1 for _, ok, _ in results if ok)
    average = sum(float(detail.rstrip("s")) for _, _, detail in results if detail.endswith("s")) / max(1, len(results))
    print("\n===== 结论 =====")
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}（{detail}）")
    print(f"\n通过 {passed}/{len(results)}｜平均单次延迟 {average:.1f}s")
    if average > 20:
        print("延迟提示：>20s/次，交互体验会很差（本系统每题要 2 次 LLM 调用：改写 + 回答）。")
    if passed < len(results):
        print("提示：不合格项会被上层闸门兜住（无引用→改判拒答），但会显著推高拒答率。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
