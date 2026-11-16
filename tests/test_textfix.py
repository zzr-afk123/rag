"""回答去噪：表格拍平、模型自带的引用清单剥离。"""

from __future__ import annotations

from rag_assistant.generation.textfix import flatten_tables, strip_source_appendix


def test_flatten_tables_turns_rows_into_sentences() -> None:
    text = (
        "第五条 单据要求\n"
        "| 费用类型 | 报销限额 | 审批人 | 办理时限 |\n"
        "|---|---|---|---|\n"
        "| 差旅住宿费（省内） | 每人每天 400 元 | 项目负责人 | 15 个工作日 |\n"
        "| 会议费 | 每人每天 500 元 | 科研处 | 20 个工作日 |"
    )
    out = flatten_tables(text)
    assert "|" not in out
    assert "第五条 单据要求" in out
    assert "费用类型：会议费；报销限额：每人每天 500 元；审批人：科研处" in out
    assert "差旅住宿费（省内）" in out


def test_flatten_tables_keeps_plain_text_untouched() -> None:
    text = "休学期一般以一年为期，累计不得超过两年。"
    assert flatten_tables(text) == text


def test_flatten_tables_header_only() -> None:
    text = "| 项目 | 金额 |\n|---|---|"
    assert flatten_tables(text) == "表格：项目；金额"


def test_strip_source_appendix_removes_model_list() -> None:
    text = (
        "会议费用的报销标准是每人每天 500 元[1]。\n"
        "引用来源 - [1] 《科研经费报销管理办法.docx》第三章 报销流程 > 第五条 单据要求"
    )
    assert strip_source_appendix(text) == "会议费用的报销标准是每人每天 500 元[1]。"


def test_strip_source_appendix_bold_heading() -> None:
    text = "教务处在综合管理中心 305[1]。\n**引用来源**\n- [1] 《办事指南》第三章"
    assert strip_source_appendix(text) == "教务处在综合管理中心 305[1]。"


def test_strip_source_appendix_keeps_normal_answer() -> None:
    text = "休学最长不得超过两年[1]。\n来源是《学生学籍管理规定》。"
    # 正文里提一句"来源是…"但后面没有引用清单，不能切
    assert strip_source_appendix(text) == text


def test_strip_source_appendix_ignores_whole_text_list() -> None:
    text = "引用来源 - [1] 《某文件》"
    assert strip_source_appendix(text) == text


def test_strip_source_appendix_inline_after_sentence() -> None:
    """模型常把清单直接接在句号后面（同一行），也要切掉。"""
    text = (
        "休学最长不得超过两年[1]。  引用来源 - [1] 《学生学籍管理规定.md》"
        "第三章 休学、复学与退学 > 第七条 休学条件（第 3 页）"
    )
    assert strip_source_appendix(text) == "休学最长不得超过两年[1]。"


def test_strip_source_appendix_inline_keeps_body_with_source_word() -> None:
    text = "材料来源：教务处窗口[1]。办理时需带学生证。"
    assert strip_source_appendix(text) == text
