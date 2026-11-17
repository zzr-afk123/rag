"""`scripts/bench_check.py` 的测试：题集与语料的对应关系必须在跑评测前校验。

合成语料是分批写的，最典型的翻车不是报错而是"静默失配"：要点不是原文逐字、
expected_doc 文件名多写一个字、章节标题跟切块器产出的对不上——这些都会让
Hit@K 和要点覆盖率变成噪声，而且数字看着还挺像回事。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import bench_check as bc


SHARED_SENTENCE = "名单公示 5 个工作日，公示期内如有异议请书面提出"


def write_doc(raw: Path, name: str, articles: dict[str, str], *, filler: str = "") -> Path:
    """按 SPEC 的结构写一篇：`## 第X章 名称` + 独占一行的 `**第X条 名称**`。

    每 3 条一章、共 3 章以上，条序号连续；`filler` 用来把篇幅顶过 SPEC 的下限，
    这样"篇幅警告"只在特意构造的用例里出现。
    """
    lines = [f"# {name.removesuffix('.md')}", "", "（2024 年 9 月印发，自 2024 年 9 月 1 日起施行）", ""]
    for index, (heading, text) in enumerate(articles.items(), start=1):
        if (index - 1) % 3 == 0:
            lines += [f"## 第{_cn((index - 1) // 3 + 1)}章 管理要求", ""]
        lines += [f"**{heading}**", "", text, ""]
    if filler:
        lines += [filler, ""]
    path = raw / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cn(number: int) -> str:
    return {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九"}.get(number, "十")


def make_golden(path: Path, items: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    filler = "本条为程序性说明，具体执行以学校最新发布的通知为准；涉及金额与时限的，以批复文件载明的数字为准。" * 30
    write_doc(
        raw,
        "学费减免办法.md",
        {
            "第一条 目的与依据": "为规范学费减免工作，根据上级有关规定，结合学校实际，制定本办法。",
            "第二条 申请条件": "家庭经济困难学生可在每学期开学后 10 个工作日内向学院提交申请表。",
            "第三条 审批时限": "学院须在收到材料后 5 个工作日内完成初审并报学生工作处。",
            "第四条 减免额度": "学费减免分三档，最高一档减免 4000 元。",
            "第五条 公示": f"{SHARED_SENTENCE}。",
            "第六条 复议": "对结果有异议可在 3 日内申请复议。",
            "第七条 材料要求": "需提交家庭经济情况表与成绩单。",
            "第八条 附则": "本办法由学生工作处负责解释。",
        },
        filler=filler,
    )
    write_doc(
        raw,
        "住宿调整办法.md",
        {
            "第一条 目的与依据": "为规范学生公寓住宿调整，制定本办法。",
            "第二条 申请窗口": "调整申请在每学期第 3 周集中受理。",
            "第三条 审批时限": "后勤处在 7 个工作日内答复。",
            "第四条 退宿手续": "退宿须归还钥匙并结清费用。",
            "第五条 押金": "宿舍钥匙押金为 40 元。",
            "第六条 检查": f"每月开展一次安全检查，{SHARED_SENTENCE}。",
            "第七条 违规处理": "违规调换床位通报学院。",
            "第八条 附则": "本办法自印发之日起施行。",
        },
        filler=filler,
    )
    return tmp_path, raw


def golden_items() -> list[dict]:
    return [
        {
            "question": "学费减免什么时候申请？",
            "expected_doc": "学费减免办法.md",
            "expected_sections": ["第二条 申请条件"],
            "expected_answer_points": ["每学期开学后 10 个工作日内"],
            "should_refuse": False,
            "variant": "literal",
            "domain": "财务",
        },
        {
            "question": "我家条件不好，想少交点学费，得走什么流程、几天能批下来？",
            "expected_doc": "学费减免办法.md",
            "expected_sections": ["第三条 审批时限"],
            "expected_answer_points": ["5 个工作日内完成初审", "最高一档减免 4000 元"],
            "should_refuse": False,
            "variant": "paraphrase",
            "domain": "财务",
        },
        {
            "question": "宿舍调换归谁管？",
            "expected_doc": "住宿调整办法.md",
            "expected_sections": ["第二条 申请窗口"],
            "expected_answer_points": ["每学期第 3 周集中受理"],
            "should_refuse": False,
            "variant": "literal",
            "domain": "后勤",
        },
        {
            "question": "学校几点停止供应热水？",
            "should_refuse": True,
            "note": "全库无热水供应时段规定",
            "variant": "paraphrase",
            "domain": "后勤",
        },
    ]


def run(tmp_path: Path, raw: Path, *files: Path, allow_unlabeled: bool = False) -> bc.Checker:
    checker = bc.Checker(raw)
    checker.load_docs()
    checker.check_goldens(list(files), allow_unlabeled=allow_unlabeled, min_doc_coverage=0.85)
    return checker


def test_valid_corpus_passes(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    golden = make_golden(tmp / "golden_X.jsonl", golden_items())
    checker = run(tmp, raw, golden)
    assert checker.errors == []
    assert checker.stats["docs"] == 2
    assert checker.stats["items"] == 4
    assert checker.stats["by_variant"] == {"literal": 2, "paraphrase": 2}
    assert checker.stats["doc_coverage"] == 1.0


def test_answer_point_must_be_verbatim(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = golden_items()
    items[0]["expected_answer_points"] = ["开学后十个工作日内"]  # 改成中文数字，原文里没有
    golden = make_golden(tmp / "golden_X.jsonl", items)
    checker = run(tmp, raw, golden)
    assert any("逐字找不到" in message for message in checker.errors)


def test_missing_doc_name_is_error(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = golden_items()
    items[0]["expected_doc"] = "学费减免管理办法.md"  # 多写"管理"两个字
    golden = make_golden(tmp / "golden_X.jsonl", items)
    checker = run(tmp, raw, golden)
    assert any("expected_doc 不存在" in message for message in checker.errors)


def test_section_must_match_article_heading(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = golden_items()
    items[1]["expected_sections"] = ["第一章 总则 > 第三条 审批时限"]  # 带路径前缀也要能认
    items[2]["expected_sections"] = ["第九条 不存在"]
    golden = make_golden(tmp / "golden_X.jsonl", items)
    checker = run(tmp, raw, golden)
    # 报错信息里会列出"可用条标题"，所以按行号判定，不能按标题子串
    assert not any(message.startswith("golden_X.jsonl:2：") for message in checker.errors)
    assert any(message.startswith("golden_X.jsonl:3：") and "第九条 不存在" in message for message in checker.errors)


def test_duplicate_question_is_error(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = golden_items()
    items.append(dict(items[0]))
    golden = make_golden(tmp / "golden_X.jsonl", items)
    checker = run(tmp, raw, golden)
    assert any("完全重复" in message for message in checker.errors)


def test_unlabeled_variant_is_error_unless_allowed(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = golden_items()
    for item in items:
        item.pop("variant", None)
        item.pop("domain", None)
    golden = make_golden(tmp / "golden_X.jsonl", items)
    assert any("缺 variant" in message for message in run(tmp, raw, golden).errors)
    assert run(tmp, raw, golden, allow_unlabeled=True).errors == []


def test_refuse_item_cannot_have_expected_doc(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = golden_items()
    items[3]["expected_doc"] = "住宿调整办法.md"
    golden = make_golden(tmp / "golden_X.jsonl", items)
    assert any("不该有 expected_doc" in message for message in run(tmp, raw, golden).errors)


def test_all_points_shared_with_other_docs_cannot_be_graded(corpus: tuple[Path, Path]) -> None:
    """要点在别的文档里也逐字出现：单个只算 warning，全中就说明这题判不了卷。"""
    tmp, raw = corpus
    items = golden_items()
    items[0]["expected_answer_points"] = [SHARED_SENTENCE]  # 学费篇与住宿篇都有这句话
    golden = make_golden(tmp / "golden_X.jsonl", items)
    checker = run(tmp, raw, golden)
    assert any("判别力弱" in message for message in checker.warnings)
    assert any("没有一个只出现在期望文档里" in message for message in checker.errors)
    # 其它题不受影响
    assert not any(message.startswith("golden_X.jsonl:2：") for message in checker.errors)


def test_untested_docs_lower_coverage_gate(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    items = [item for item in golden_items() if item.get("expected_doc") != "住宿调整办法.md"]
    golden = make_golden(tmp / "golden_X.jsonl", items)
    checker = run(tmp, raw, golden)
    assert any("覆盖率" in message for message in checker.errors)
    assert any("没有题目引用" in message for message in checker.warnings)


def test_structure_and_numbering_checks(corpus: tuple[Path, Path]) -> None:
    tmp, raw = corpus
    short = raw / "太短办法.md"
    short.write_text("# 太短办法\n\n## 第一章 总则\n\n**第一条 目的**\n\n一句话。\n", encoding="utf-8")
    golden = make_golden(tmp / "golden_X.jsonl", golden_items())
    checker = bc.Checker(raw)
    checker.load_docs()
    assert any("只有 1 个" in message for message in checker.errors)
    assert any("篇幅" in message for message in checker.warnings)
    short.unlink()

    broken = raw / "序号断档办法.md"
    body = "\n".join(
        [
            "# 序号断档办法",
            "",
            "## 第一章 总则",
            "",
            "**第一条 目的**",
            "",
            "为规范本事项，制定本办法。" * 60,
            "",
            "**第三条 期限**",
            "",
            "应当在 10 个工作日内办结。",
        ]
    )
    broken.write_text(body, encoding="utf-8")
    checker = bc.Checker(raw)
    checker.load_docs()
    assert any("条款序号不连续" in message for message in checker.errors)


def test_cn_number_to_int() -> None:
    assert bc.cn_number_to_int("第七条 五双管理") == 7
    assert bc.cn_number_to_int("第十二条 罚则") == 12
    assert bc.cn_number_to_int("第二十三条 附则") == 23
    assert bc.cn_number_to_int("第一百零三条") is None
    assert bc.cn_number_to_int("第 3 条") is None


def test_normalize_collapses_spaces_and_fullwidth() -> None:
    assert bc.normalize("减免 4000　元") == bc.normalize("减免4000元")


def test_cli_reports_errors_and_exit_code(corpus: tuple[Path, Path], capsys: pytest.CaptureFixture) -> None:
    tmp, raw = corpus
    items = golden_items()
    items[0]["expected_answer_points"] = ["不存在的话"]
    bad = make_golden(tmp / "golden_X.jsonl", items)
    assert bc.main(["--raw", str(raw), "--golden", str(bad)]) == 1
    printed = capsys.readouterr().out
    assert "ERROR" in printed and "逐字找不到" in printed

    good = make_golden(tmp / "golden_Y.jsonl", golden_items())
    assert bc.main(["--raw", str(raw), "--golden", str(good)]) == 0
