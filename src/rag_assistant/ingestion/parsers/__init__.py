"""解析器注册入口。

导入本包即完成注册（各模块用 ``@register_parser`` 装饰器自动登记）。
"""

from __future__ import annotations

from .base import (
    BaseParser,
    SectionBuilder,
    get_parser,
    guess_heading,
    register_parser,
    supported_extensions,
)
from .docx_parser import DocxParser
from .md_parser import MarkdownParser
from .pdf_parser import PdfParser
from .txt_parser import TxtParser
from .xlsx_parser import XlsxParser

__all__ = [
    "BaseParser",
    "DocxParser",
    "MarkdownParser",
    "PdfParser",
    "SectionBuilder",
    "TxtParser",
    "XlsxParser",
    "get_parser",
    "guess_heading",
    "register_parser",
    "supported_extensions",
]
