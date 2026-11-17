"""对抗评测：用一批"不好好问"的问题检验系统的边界。

和 ``evaluate.py`` 的 36 题黄金集不同，这里故意换说法、打错字、给错误前提、
问知识库之外的事——**目标不是拿高分，而是把边界摆在明面上**：

- 近义改写（paraphrase）：换个说法还能不能检索到；
- 错别字（typo）：用户真实输入里常见的手误；
- 跨文档（cross_doc）：同一事实分散在多份文件里；
- 诱导编造（induced）：问题里塞一个错误前提，看模型是否顺着说；
- 知识库外（off_topic）：该拒答的是否拒得干脆。

产出三项关键指标：

1. **拒答判断准确率**：该答的答、该拒的拒；
2. **误拒率**：能答的问题被拒掉的比例（对体验伤害最大）；
3. **被诱导成功率**：错误前提被顺着接受的比例（对可信度伤害最大）。

用法（服务需先启动，默认连本机 8000）::

    python scripts/eval_adversarial.py
    python scripts/eval_adversarial.py --base http://127.0.0.1:8000 --limit 5
    python scripts/eval_adversarial.py --out reports/adversarial.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = PROJECT_ROOT / "data" / "eval" / "adversarial_set.jsonl"
KIND_LABELS = {
    "paraphrase": "近义改写",
    "typo": "错别字",
    "cross_doc": "跨文档",
    "induced": "诱导编造",
    "off_topic": "知识库外",
}


@dataclass
class Case:
    question: str
    kind: str = "paraphrase"
    expected_doc: str = ""
    expected_docs: list[str] = field(default_factory=list)
    expected_answer_points: list[str] = field(default_factory=list)
    forbidden_points: list[str] = field(default_factory=list)
    should_refuse: bool = False
    note: str = ""

    @property
    def docs(self) -> list[str]:
        names = [name for name in self.expected_docs if name]
        if self.expected_doc:
            names.append(self.expected_doc)
        return names


@dataclass
class Result:
    case: Case
    answer: str
    refused: bool
    doc_names: list[str]
    latency_ms: float
    error: str = ""

    @property
    def decision_ok(self) -> bool:
        return self.refused == self.case.should_refuse

    @property
    def doc_hit(self) -> bool:
        if not self.case.docs:
            return False
        joined = " ".join(self.doc_names)
        return any(name in joined for name in self.case.docs)

    @property
    def point_hits(self) -> dict[str, bool]:
        text = self.answer.replace(" ", "")
        return {point: point.replace(" ", "") in text for point in self.case.expected_answer_points}

    @property
    def induced(self) -> bool:
        """错误前提被接受：出现了错误说法，又没有给出正确要点。"""
        if not self.case.forbidden_points:
            return False
        text = self.answer.replace(" ", "")
        bad = any(point.replace(" ", "") in text for point in self.case.forbidden_points)
        good = any(self.point_hits.values())
        return bad and not good


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number} JSON 解析失败：{exc}") from exc
        if not payload.get("question"):
            raise SystemExit(f"{path}:{number} 缺少 question")
        cases.append(
            Case(
                question=payload["question"],
                kind=payload.get("kind", "paraphrase"),
                expected_doc=payload.get("expected_doc", ""),
                expected_docs=list(payload.get("expected_docs") or []),
                expected_answer_points=list(payload.get("expected_answer_points") or []),
                forbidden_points=list(payload.get("forbidden_points") or []),
                should_refuse=bool(payload.get("should_refuse", False)),
                note=payload.get("note", ""),
            )
        )
    return cases


def ask(client, base: str, question: str, session_id: str) -> Result:
    payload = {"question": question, "session_id": session_id}
    started = time.perf_counter()
    try:
        resp = client.post(f"{base}/chat", json=payload, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - 单题失败不该中断整轮评测
        case = Case(question=question)
        return Result(
            case=case,
            answer="",
            refused=False,
            doc_names=[],
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency = float(data.get("latency_ms") or (time.perf_counter() - started) * 1000)
    citations = data.get("citations") or []
    return Result(
        case=Case(question=question),
        answer=str(data.get("answer") or ""),
        refused=bool(data.get("refused")),
        doc_names=[str(item.get("doc_name") or "") for item in citations],
        latency_ms=latency,
    )


def compute(results: list[Result]) -> dict[str, Any]:
    answerable = [r for r in results if not r.case.should_refuse]
    refusable = [r for r in results if r.case.should_refuse]
    induced = [r for r in results if r.case.forbidden_points]

    def rate(hit: int, total: int) -> float:
        return round(hit / total, 4) if total else 0.0

    latencies = sorted(r.latency_ms for r in results if not r.error)
    point_hits = [hit for r in answerable for hit in r.point_hits.values()]
    return {
        "total": len(results),
        "errors": sum(1 for r in results if r.error),
        "decision_accuracy": rate(sum(1 for r in results if r.decision_ok), len(results)),
        "answered": sum(1 for r in results if not r.refused),
        "refused": sum(1 for r in results if r.refused),
        "false_refusal": sum(1 for r in answerable if r.refused),
        "false_refusal_rate": rate(sum(1 for r in answerable if r.refused), len(answerable)),
        "refusal_recall": rate(sum(1 for r in refusable if r.refused), len(refusable)),
        "doc_hit_rate": rate(sum(1 for r in answerable if r.doc_hit), len(answerable)),
        "point_coverage": rate(sum(1 for hit in point_hits if hit), len(point_hits)),
        "induced_total": len(induced),
        "induced_accepted": sum(1 for r in induced if r.induced),
        "induced_rate": rate(sum(1 for r in induced if r.induced), len(induced)),
        "latency_p50": round(statistics.median(latencies) / 1000, 2) if latencies else 0.0,
        "latency_p95": round(latencies[max(0, int(len(latencies) * 0.95) - 1)] / 1000, 2)
        if latencies
        else 0.0,
    }


def by_kind(results: list[Result]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Result]] = {}
    for result in results:
        groups.setdefault(result.case.kind, []).append(result)
    payload: dict[str, dict[str, Any]] = {}
    for kind, rows in groups.items():
        answerable = [r for r in rows if not r.case.should_refuse]
        payload[kind] = {
            "total": len(rows),
            "decision_accuracy": round(sum(1 for r in rows if r.decision_ok) / len(rows), 4),
            "doc_hit_rate": round(
                sum(1 for r in answerable if r.doc_hit) / len(answerable), 4
            )
            if answerable
            else 0.0,
            "false_refusal": sum(1 for r in answerable if r.refused),
            "induced": sum(1 for r in rows if r.induced),
        }
    return payload


def render(results: list[Result], metrics: dict[str, Any], *, base: str) -> str:
    lines: list[str] = []
    lines.append("# 对抗评测报告")
    lines.append("")
    lines.append(f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 服务：{base}")
    lines.append(f"- 题目：{metrics['total']} 道（近义改写 / 错别字 / 跨文档 / 诱导编造 / 知识库外）")
    lines.append("")
    lines.append("## 三项关键指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 拒答判断准确率 | {metrics['decision_accuracy']:.1%} | 该答的答、该拒的拒 |"
    )
    lines.append(
        f"| 误拒率 | {metrics['false_refusal_rate']:.1%} "
        f"（{metrics['false_refusal']} 题） | 能答的问题被拒掉 |"
    )
    lines.append(
        f"| 被诱导成功率 | {metrics['induced_rate']:.1%} "
        f"（{metrics['induced_accepted']}/{metrics['induced_total']}） | 顺着错误前提说 |"
    )
    lines.append("")
    lines.append("## 辅助指标")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 文档命中率 | {metrics['doc_hit_rate']:.1%} |")
    lines.append(f"| 要点覆盖率 | {metrics['point_coverage']:.1%} |")
    lines.append(f"| 该拒的拒掉 | {metrics['refusal_recall']:.1%} |")
    lines.append(f"| 延迟 P50 / P95 | {metrics['latency_p50']}s / {metrics['latency_p95']}s |")
    lines.append("")
    lines.append("## 分类表现")
    lines.append("")
    lines.append("| 类别 | 题数 | 判断准确率 | 文档命中 | 误拒 | 被诱导 |")
    lines.append("|---|---|---|---|---|---|")
    for kind, row in by_kind(results).items():
        lines.append(
            f"| {KIND_LABELS.get(kind, kind)} | {row['total']} | {row['decision_accuracy']:.3f} | "
            f"{row['doc_hit_rate']:.3f} | {row['false_refusal']} | {row['induced']} |"
        )
    lines.append("")
    lines.append("## 逐题明细")
    lines.append("")
    lines.append("| # | 类别 | 问题 | 判断 | 引用 | 要点 | 耗时 | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for index, result in enumerate(results, start=1):
        case = result.case
        decision = "✅" if result.decision_ok else ("❌ 误拒" if case.should_refuse is False else "❌ 该拒不拒")
        if result.error:
            decision = "⚠️ 调用失败"
        doc = "—" if case.should_refuse or not case.docs else ("✅" if result.doc_hit else "❌")
        hits = result.point_hits
        points = "—" if not hits else ("✅" if all(hits.values()) else "⚠️ " + "/".join(k for k, v in hits.items() if not v) + " 缺失")
        note = case.note or case.question
        if result.induced:
            note += "（被诱导）"
        lines.append(
            f"| {index} | {KIND_LABELS.get(case.kind, case.kind)} | {case.question} | {decision} | "
            f"{doc} | {points} | {result.latency_ms / 1000:.1f}s | {note} |"
        )
        if result.error:
            lines.append(f"| | | 错误：{result.error} | | | | | |")
    lines.append("")
    lines.append("## 怎么读这份报告")
    lines.append("")
    lines.append("- **误拒率**比「命中率」更伤体验：用户问得对却被拒，会直接放弃使用；")
    lines.append("- **被诱导成功率**是可信度的底线：问题是错的，答案应该纠正而不是顺着说；")
    lines.append("- 错别字题失手说明检索对字面敏感，可以补同义词或加一层纠错；")
    lines.append("- 跨文档题失手说明同一事实散落多份文件，需要考虑合并或加交叉引用。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对抗评测（改写 / 错别字 / 跨文档 / 诱导 / 越界）")
    parser.add_argument("--set", type=Path, default=DEFAULT_SET, help="题目文件（JSONL）")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="服务地址")
    parser.add_argument("--user", default="20230001", help="登录账号")
    parser.add_argument("--password", default="student123", help="登录密码")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--out", type=Path, default=None, help="报告路径")
    parser.add_argument("--json", action="store_true", help="把摘要打印成 JSON")
    args = parser.parse_args(argv)

    try:
        import httpx
    except ModuleNotFoundError:
        print("需要 httpx：pip install httpx", file=sys.stderr)
        return 2

    # Windows 控制台默认 GBK，打印中文/表情会抛 UnicodeEncodeError
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - 老版本 Python 没有 reconfigure
        pass

    cases = load_cases(args.set)
    if args.limit:
        cases = cases[: args.limit]

    with httpx.Client(base_url=args.base, timeout=180.0) as client:
        login = client.post("/auth/login", json={"id": args.user, "password": args.password})
        if login.status_code != 200:
            guest = client.post("/auth/guest")
            if guest.status_code != 200:
                print(f"登录失败：{login.status_code} {login.text}", file=sys.stderr)
                return 2
            print("示例账号不可用，改用访客身份评测（只影响会话归属）", file=sys.stderr)

        results: list[Result] = []
        for index, case in enumerate(cases, start=1):
            result = ask(client, args.base, case.question, f"adv-{int(time.time())}-{index}")
            result.case = case
            results.append(result)
            flag = "OK" if result.decision_ok else "!!"
            print(f"[{index:2d}/{len(cases)}] {flag} {result.latency_ms / 1000:5.1f}s {case.question}")

    metrics = compute(results)
    report = render(results, metrics, base=args.base)
    out_path = args.out or (
        PROJECT_ROOT / "reports" / f"eval-adversarial-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print("")
    print(f"报告已写入：{out_path}")
    print(
        f"拒答判断准确率={metrics['decision_accuracy']:.1%} "
        f"误拒率={metrics['false_refusal_rate']:.1%} "
        f"被诱导成功率={metrics['induced_rate']:.1%} "
        f"文档命中={metrics['doc_hit_rate']:.1%} "
        f"要点覆盖={metrics['point_coverage']:.1%}"
    )
    if args.json:
        print(json.dumps({"metrics": metrics, "by_kind": by_kind(results)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
