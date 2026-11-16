"""生成二进制格式的示例文档（DOCX / XLSX）。

为什么用脚本而不是直接把二进制文件塞进仓库：
1. 二进制不可 diff，review 时看不出内容；
2. 脚本本身是可读的"文档结构说明"，同时验证解析器对真实文件的行为。

用法::

    python scripts/gen_sample_docs.py            # 写入 data/raw/
    python scripts/gen_sample_docs.py --out tmp  # 写入自定义目录

内容为虚构的示例制度，仅用于演示与测试，与真实院校制度无关。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.shared import Pt
from openpyxl import Workbook
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw"


def build_docx(target: Path) -> Path:
    """《科研经费报销管理办法》：带多级标题 + 一张表格。"""
    doc = Document()
    doc.styles["Normal"].font.name = "宋体"
    doc.styles["Normal"].font.size = Pt(12)

    doc.add_heading("郑州大学科研经费报销管理办法", level=1)
    doc.add_paragraph("（2024 年修订，自 2024 年 7 月 1 日起施行）")

    doc.add_heading("第一章 总则", level=2)
    doc.add_heading("第一条 适用范围", level=3)
    doc.add_paragraph(
        "本办法适用于学院在编教职工使用财政性科研经费、横向课题经费、校级科研基金"
        "所发生的差旅费、会议费、材料费、测试化验加工费、劳务费等支出报销。"
    )

    doc.add_heading("第二条 管理职责", level=3)
    doc.add_paragraph("科研处负责经费预算与结题审核，财务处负责票据审核与付款，项目负责人对经费使用的真实性与相关性负责。")

    doc.add_heading("第二章 报销标准", level=2)
    doc.add_heading("第三条 差旅费", level=3)
    doc.add_paragraph(
        "出差人员应在出差结束后 15 个工作日内办理报销。城市间交通费凭据报销，"
        "住宿费按标准限额内据实报销，超出部分由个人承担。"
    )

    doc.add_heading("第四条 劳务费", level=3)
    doc.add_paragraph(
        "劳务费发放对象为参与项目研究的学生及校外人员，不得发放给项目组成员中的在编教职工。"
        "单次发放金额超过 800 元的，须附工作量说明并由项目负责人签字确认。"
    )

    doc.add_heading("第三章 报销流程", level=2)
    doc.add_heading("第五条 单据要求", level=3)
    doc.add_paragraph("报销须提供合规发票原件、审批单、经费卡复印件；单张发票金额超过 5000 元的，须附合同或采购比价记录。")

    table = doc.add_table(rows=4, cols=4)
    table.style = "Table Grid"
    headers = ["费用类型", "报销限额", "审批人", "办理时限"]
    for index, text in enumerate(headers):
        table.cell(0, index).text = text
    rows = [
        ["差旅住宿费（省内）", "每人每天 400 元", "项目负责人", "15 个工作日"],
        ["会议费", "每人每天 500 元", "科研处", "20 个工作日"],
        ["材料费", "据实报销", "项目负责人", "10 个工作日"],
    ]
    for row_index, row in enumerate(rows, start=1):
        for col_index, text in enumerate(row):
            table.cell(row_index, col_index).text = text

    doc.add_heading("第四章 附则", level=2)
    doc.add_paragraph("本办法由财务处会同科研处负责解释，自发布之日起施行。")

    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(target))
    return target


def build_xlsx(target: Path) -> Path:
    """《差旅与会议费标准表》：两个 sheet，供表格整体保留逻辑测试。"""
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "差旅住宿费标准"
    header = ["地区类别", "职务/职称", "住宿费限额（元/天）", "伙食补助（元/天）", "市内交通（元/天）"]
    sheet.append(header)
    data = [
        ["省内", "教授及相当职务", 500, 100, 80],
        ["省内", "其他人员", 400, 100, 80],
        ["省外一般地区", "教授及相当职务", 600, 100, 80],
        ["省外一般地区", "其他人员", 500, 100, 80],
        ["省外特殊地区", "不限", 800, 120, 100],
    ]
    for row in data:
        sheet.append(row)
    for cell in sheet[1]:
        cell.font = Font(bold=True)

    sheet2 = workbook.create_sheet("会议费标准")
    sheet2.append(["会议类别", "住宿费（元/人天）", "伙食费（元/人天）", "场地租金（元/天）"])
    for row in [
        ["院内会议", 0, 80, 500],
        ["省内学术会议", 450, 130, 2000],
        ["全国性学术会议", 600, 150, 4000],
    ]:
        sheet2.append(row)
    for cell in sheet2[1]:
        cell.font = Font(bold=True)

    for current in workbook.worksheets:
        for column in "ABCDE":
            current.column_dimensions[column].width = 20

    target.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(str(target))
    return target


def build_lab_docx(target: Path) -> Path:
    """实验室安全管理规定（DOCX）：用于覆盖"安全责任/危化品/事故上报"这类主题。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_heading("郑州大学实验室安全管理规定", level=1)

    document.add_heading("第一章 总则", level=2)
    document.add_heading("第一条 目的与依据", level=3)
    document.add_paragraph(
        "为保障师生人身安全与实验室财产安全，依据《高等学校实验室安全规范》，制定本规定。"
    )
    document.add_heading("第二条 适用范围", level=3)
    document.add_paragraph(
        "本规定适用于学院所有教学实验室、科研实验室及公共测试平台，含临时借用的实验场所。"
    )

    document.add_heading("第二章 安全责任", level=2)
    document.add_heading("第三条 责任体系", level=3)
    document.add_paragraph(
        "实验室安全实行院、二级单位、实验室三级管理。实验室与设备管理处负责监督检查，"
        "二级单位为直接责任单位，实验室负责人为第一责任人。"
    )
    document.add_heading("第四条 导师与指导教师责任", level=3)
    document.add_paragraph(
        "研究生导师、课程指导教师对学生实验安全负直接管理责任，须在实验前完成安全交底并留存记录。"
    )

    document.add_heading("第三章 危险化学品管理", level=2)
    document.add_heading("第五条 采购与登记", level=3)
    document.add_paragraph(
        "管制类危险化学品须经实验室与设备管理处审批后统一采购，入库后 3 个工作日内完成台账登记，"
        "做到账物相符，每学期末盘点一次。"
    )
    document.add_heading("第六条 存放要求", level=3)
    document.add_paragraph("危险化学品按类别分区存放，禁止混放；存放场所应通风、避光，并配备消防器材。")
    table = document.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    header = table.rows[0].cells
    header[0].text = "类别"
    header[1].text = "存放条件"
    header[2].text = "单间最大存量"
    for category, condition, limit in [
        ("易燃液体", "防爆柜，远离热源", "20 升"),
        ("强酸强碱", "防腐柜，与有机物隔离", "10 升"),
        ("剧毒品", "双人双锁保险柜", "500 克"),
        ("压缩气体", "气瓶架固定，直立存放", "6 瓶"),
    ]:
        cells = table.add_row().cells
        cells[0].text = category
        cells[1].text = condition
        cells[2].text = limit

    document.add_heading("第四章 实验操作与防护", level=2)
    document.add_heading("第七条 个人防护", level=3)
    document.add_paragraph(
        "进入实验室须穿实验服、佩戴护目镜；接触挥发性有机溶剂时须在通风橱内操作。"
        "长发须束起，禁止穿拖鞋、短裤进入实验室。"
    )
    document.add_heading("第八条 高危实验审批", level=3)
    document.add_paragraph(
        "涉及高温高压、剧毒、放射性的实验须提前 5 个工作日提交风险评估报告，"
        "经二级单位审核、实验室与设备管理处批准后方可实施，实验时须两人以上在场。"
    )

    document.add_heading("第五章 事故处理", level=2)
    document.add_heading("第九条 报告时限", level=3)
    document.add_paragraph(
        "发生实验室安全事故，现场人员应立即处置并同时报告实验室负责人；"
        "实验室负责人须在 1 小时内报实验室与设备管理处，不得迟报、瞒报。"
    )
    document.add_heading("第十条 处置流程", level=3)
    document.add_paragraph(
        "先救人、后处置：切断电源与气源、疏散人员、隔离现场；涉及人员受伤的，第一时间拨打校医院电话并送医。"
    )

    document.add_heading("第六章 附则", level=2)
    document.add_heading("第十一条 解释权", level=3)
    document.add_paragraph("本规定由实验室与设备管理处负责解释。")
    document.add_heading("第十二条 施行日期", level=3)
    document.add_paragraph("本规定自 2024 年 11 月 1 日起施行。")

    document.save(str(target))
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="生成示例 DOCX/XLSX 文档")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出目录")
    args = parser.parse_args()

    out_dir: Path = args.out
    for path in (
        build_docx(out_dir / "科研经费报销管理办法.docx"),
        build_xlsx(out_dir / "差旅与会议费标准表.xlsx"),
        build_lab_docx(out_dir / "实验室安全管理规定.docx"),
    ):
        print(f"已生成：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
