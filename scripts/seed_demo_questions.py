"""生成一批演示问答日志，让部门看板/周报在没真实流量时也能看效果。

**只写日志，不调 LLM、不碰索引**，所以几秒钟就能跑完。

用法::

    python scripts\\seed_demo_questions.py            # 默认 60 条，摊在近 30 天
    python scripts\\seed_demo_questions.py --count 120 --days 14
    python scripts\\seed_demo_questions.py --clear    # 清掉演示数据（只删 demo 字段的记录）

注意：
1. 记录带 ``"demo": true`` 标记，看板会提示"含演示数据"；
2. 写进的是真实的 ``logs/qa.jsonl``，上线前用 ``--clear`` 或直接删 ``logs/qa.jsonl*``；
3. 问题取自真实制度语料的常见问法，部门由关键词自动命中（与线上同一套逻辑）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_assistant.config import load_settings  # noqa: E402
from rag_assistant.departments import DepartmentDirectory  # noqa: E402

# (问题, 是否容易被答上) —— 后者为 False 的用来模拟"知识库还没覆盖"的缺口
QUESTIONS: list[tuple[str, bool]] = [
    ("休学最长可以多久？", True),
    ("复学需要提交什么材料？", True),
    ("新生请假最多可以请多久？", True),
    ("每学期什么时候注册？没按时注册会怎样？", True),
    ("哪些情况会被退学？", True),
    ("家里经济困难，能先注册后补学费吗？", True),
    ("学士学位由谁审议授予？", True),
    ("在读证明去哪里办？", True),
    ("线上申请在读证明多久能办完？", True),
    ("同一学期重复开在读证明要收钱吗？", True),
    ("让同学替我去开在读证明，需要带什么？", True),
    ("毕业以后还能开在读证明吗？", True),
    ("省内出差住宿费每人每天限额是多少？", True),
    ("会议费每人每天限额多少？由谁审批？", True),
    ("差旅费报销要在多少天内办理？", True),
    ("单张发票金额超过 5000 元，报销时还要附什么？", True),
    ("劳务费报销需要哪些附件？", True),
    ("图书馆周五几点开门？", True),
    ("周末图书馆几点关门？", True),
    ("本科生一次最多能借几本书？借多久？", True),
    ("图书逾期一天罚多少钱？", True),
    ("在校外怎么用学校的电子数据库？", True),
    ("研讨间一次最多能用多久？怎么约？", True),
    ("自习座位一次能约几个小时？", True),
    ("危险化学品入库后多久要完成台账登记？", True),
    ("剧毒品应该怎么存放？", True),
    ("易燃液体在实验室单间最多能存多少？", True),
    ("实验室出事故后多久必须上报？", True),
    ("做高危实验要提前几天提交风险评估？", True),
    ("进实验室对着装有什么要求？", True),
    # 以下几条用来体现"知识库缺口"与"识别不到部门"
    ("研究生国家奖学金什么时候评？", False),
    ("交换生的学分怎么认定？", False),
    ("辅修第二专业的学费怎么算？", False),
    ("宿舍空调坏了找谁修？", False),
    ("校园卡丢了怎么补办？", False),
    ("食堂几点开饭？", False),
]


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成演示问答日志（部门看板用）")
    parser.add_argument("--count", type=int, default=60, help="生成多少条记录")
    parser.add_argument("--days", type=int, default=30, help="摊在最近多少天里")
    parser.add_argument("--seed", type=int, default=20261111, help="随机种子（可复现）")
    parser.add_argument("--clear", action="store_true", help="删除所有 demo=true 的记录")
    args = parser.parse_args(argv)

    settings = load_settings(ROOT / "config" / "settings.yaml", root=ROOT)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    target = settings.log_dir / settings.logging.qa_log
    directory = DepartmentDirectory.from_settings(settings)

    existing = _load_existing(target)
    if args.clear:
        kept = [row for row in existing if not row.get("demo")]
        removed = len(existing) - len(kept)
        with target.open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"已删除 {removed} 条演示记录，保留 {len(kept)} 条真实记录")
        return 0

    rng = random.Random(args.seed)
    now = datetime.now()
    rows: list[dict] = []
    for _ in range(max(1, args.count)):
        question, answerable = rng.choice(QUESTIONS)
        # 时间摊开：越近的日期权重略高，看起来更像真实使用
        offset = int(abs(rng.gauss(0, args.days / 2.2))) % max(1, args.days)
        moment = now - timedelta(days=offset, hours=rng.randint(0, 8), minutes=rng.randint(0, 59))
        refused = (not answerable) or rng.random() < 0.08
        depts = [item.id for item in directory.match(question)]
        rows.append(
            {
                "ts": moment.isoformat(timespec="seconds"),
                "session_id": f"demo-{rng.randint(1000, 9999)}",
                "question": question,
                "rewritten_query": question,
                "retrieved_ids": [],
                "latency_ms": round(rng.uniform(1200, 3200), 1),
                "refused": refused,
                "refusal_reason": "below_threshold" if refused else None,
                "departments": depts,
                "demo": True,
            }
        )
    rows.sort(key=lambda item: item["ts"])

    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_dept: dict[str, int] = {}
    for row in rows:
        for dept_id in row["departments"]:
            by_dept[dept_id] = by_dept.get(dept_id, 0) + 1
    print(f"已写入 {len(rows)} 条演示记录 → {target}")
    print("部门分布：" + "、".join(f"{directory.by_id(k).name if directory.by_id(k) else k}={v}" for k, v in sorted(by_dept.items(), key=lambda kv: -kv[1])))
    print(f"未答上：{sum(1 for row in rows if row['refused'])} 条")
    print("打开 /dept 查看；清理用 --clear")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
