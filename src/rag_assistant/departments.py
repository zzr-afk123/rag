"""部门目录：关键词 → 部门 → 联系方式。

三件事，刻意保持"笨"：

1. **纯关键词命中**，不做语义推断。问"我生病了想休息一学期"不会命中教务处，
   但会命中"休学"这个同义词——所以关键词表里要放制度用语，而不是只放部门名。
2. **只提示，不改写答案**。联系方式是结构化字段（`AnswerResult.contacts`），
   由前端渲染成卡片；如果塞进 answer 正文，会污染忠实度评测，也会让提示看起来
   像模型自己编的。
3. **配置驱动**。部门、电话、邮箱、办公地点全在 `config/departments.yaml`，
   换学校只改配置；文件缺失时整个功能静默降级（不提示，不报错）。

命中阈值：至少 1 个关键词。同一部门命中多个关键词只算一次（按命中数排序，
命中越多越靠前），最多返回 `departments.max_suggestions` 个。

为什么"提示部门"这件事值得做：制度问答里"该找谁"和"规定是什么"同样重要。
用户拿到答案（甚至被拒答）后，最需要的往往是**下一步打哪个电话**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from .logging_setup import get_logger

logger = get_logger("departments")


@dataclass(frozen=True)
class Department:
    """一个可被追问的部门。"""

    id: str
    name: str
    full_name: str = ""
    keywords: tuple[str, ...] = ()
    phone: str = ""
    email: str = ""
    address: str = ""
    hours: str = ""
    note: str = ""

    @property
    def title(self) -> str:
        return self.full_name or self.name

    @property
    def has_contact(self) -> bool:
        """至少有电话或邮箱才值得提示（否则只是"去找某部门"的空话）。"""
        return bool(self.phone.strip() or self.email.strip())

    def to_dict(self) -> dict[str, object]:
        """对外结构（前端直接渲染；不含关键词，避免把内部匹配规则暴露给用户）。"""
        return {
            "id": self.id,
            "name": self.name,
            "full_name": self.full_name,
            "phone": self.phone,
            "email": self.email,
            "address": self.address,
            "hours": self.hours,
            "note": self.note,
        }


@dataclass(frozen=True)
class DepartmentHit:
    """一次命中的结果（带分数，便于调试"为什么提示了它"）。"""

    department: Department
    score: int
    keywords: tuple[str, ...] = ()


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable):
        items = [str(item) for item in value]
    else:  # pragma: no cover - YAML 类型异常
        return ()
    return tuple(item.strip() for item in items if item and str(item).strip())


@dataclass
class DepartmentDirectory:
    """部门目录（加载一次，进程内复用）。"""

    departments: tuple[Department, ...] = ()
    max_suggestions: int = 3
    source: str = ""
    _index: dict[str, Department] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {department.id: department for department in self.departments}

    # ---- 加载 ----
    @classmethod
    def load(cls, path: Path | str | None, *, max_suggestions: int = 3) -> "DepartmentDirectory":
        """从 YAML 加载；文件不存在/格式错误都退化为空目录（不抛）。"""
        if not path:
            return cls(max_suggestions=max_suggestions)
        target = Path(path)
        if not target.exists():
            logger.info("部门目录不存在（跳过联系方式提示）：%s", target)
            return cls(max_suggestions=max_suggestions, source=str(target))
        try:
            raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - 配置坏了不该让服务起不来
            logger.warning("部门目录解析失败（跳过联系方式提示）：%s", exc)
            return cls(max_suggestions=max_suggestions, source=str(target))

        items = raw.get("departments") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            logger.warning("部门目录格式不是列表（%s），已忽略", target)
            return cls(max_suggestions=max_suggestions, source=str(target))

        departments: list[Department] = []
        seen: set[str] = set()
        for entry in items:
            if not isinstance(entry, dict):
                continue
            dept_id = str(entry.get("id") or entry.get("name") or "").strip()
            name = str(entry.get("name") or "").strip()
            if not dept_id or not name or dept_id in seen:
                logger.warning("部门条目缺少 id/name 或 id 重复，已跳过：%r", entry)
                continue
            seen.add(dept_id)
            departments.append(
                Department(
                    id=dept_id,
                    name=name,
                    full_name=str(entry.get("full_name") or "").strip(),
                    keywords=_as_tuple(entry.get("keywords")),
                    phone=str(entry.get("phone") or "").strip(),
                    email=str(entry.get("email") or "").strip(),
                    address=str(entry.get("address") or "").strip(),
                    hours=str(entry.get("hours") or "").strip(),
                    note=str(entry.get("note") or "").strip(),
                )
            )
        logger.info("部门目录已加载：%d 个部门（%s）", len(departments), target)
        return cls(
            departments=tuple(departments),
            max_suggestions=max(1, int(max_suggestions)),
            source=str(target),
        )

    @classmethod
    def from_settings(cls, settings) -> "DepartmentDirectory":
        config = getattr(settings, "departments", None)
        if config is not None and not getattr(config, "enabled", True):
            return cls()
        path = getattr(settings, "departments_file", None)
        return cls.load(path, max_suggestions=getattr(config, "max_suggestions", 3) if config else 3)

    # ---- 查询 ----
    @property
    def enabled(self) -> bool:
        return bool(self.departments)

    def by_id(self, dept_id: str) -> Department | None:
        return self._index.get(dept_id)

    def score(self, text: str) -> list[DepartmentHit]:
        """对一段文本打分（命中关键词数；命中部门全名额外 +2）。"""
        if not text:
            return []
        hits: list[DepartmentHit] = []
        for department in self.departments:
            matched = tuple(
                keyword
                for keyword in department.keywords
                if keyword and keyword in text
            )
            bonus = 0
            if department.name and department.name in text and department.name not in matched:
                bonus = 2
            if not matched and not bonus:
                continue
            hits.append(
                DepartmentHit(
                    department=department,
                    score=len(matched) + bonus,
                    keywords=matched,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def match(
        self,
        question: str,
        *,
        rewritten: str = "",
        doc_names: Sequence[str] = (),
        only_with_contact: bool = True,
    ) -> list[Department]:
        """综合"问题 + 改写 + 引用文档名"给出建议部门。

        权重：问题命中 2 分、改写命中的只有问题没命中的补 1 分、
        引用文档名命中 1 分（文档名常含"学籍""报销"这类部门特征词）。
        这样"问休学"（问题命中教务处）不会被"引用了一份财务表格"挤掉。
        """
        if not self.departments:
            return []
        scores: dict[str, int] = {}
        order: dict[str, int] = {}
        for position, department in enumerate(self.departments):
            order[department.id] = position

        def add(text: str, weight: int) -> None:
            for hit in self.score(text):
                scores[hit.department.id] = scores.get(hit.department.id, 0) + hit.score * weight

        add(question, 2)
        if rewritten and rewritten != question:
            add(rewritten, 1)
        for name in doc_names:
            add(name, 1)

        ranked = sorted(
            (dept_id for dept_id in scores if scores[dept_id] > 0),
            key=lambda dept_id: (-scores[dept_id], order[dept_id]),
        )
        result: list[Department] = []
        for dept_id in ranked:
            department = self._index[dept_id]
            if only_with_contact and not department.has_contact:
                continue
            result.append(department)
            if len(result) >= self.max_suggestions:
                break
        return result

    def contacts_for(
        self,
        question: str,
        *,
        rewritten: str = "",
        doc_names: Sequence[str] = (),
    ) -> list[dict[str, object]]:
        """给 API / SSE 用的结构（直接 JSON 序列化）。"""
        return [
            department.to_dict()
            for department in self.match(question, rewritten=rewritten, doc_names=doc_names)
        ]

    def catalog(self) -> list[dict[str, object]]:
        """部门清单（看板筛选用，不含关键词）。"""
        return [department.to_dict() for department in self.departments]


__all__ = ["Department", "DepartmentDirectory", "DepartmentHit"]
