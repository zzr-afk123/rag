"""查询归一：把口语说法换回公文用词。

用户问"我想进校访问"，制度原文写的是"入校"与"访客"——分词之后 ``进校`` 与
``入校`` 是两个毫不相干的词条，BM25 一分不给，词面覆盖闸门也会把它算成"没命中"。
这不是模型的错，是词表的错。

这里维护一张"同义词 → 公文词"的表（``config/synonyms.yaml``），检索前先把问句里
的口语说法替换掉，让三条链路同时受益：向量召回不再被措辞带偏、BM25 能命中、
覆盖闸门算得准。

替换按词长从长到短进行，避免"进校园"被"进校"切一半；同一位置只替换一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class SynonymTable:
    """一张不可变的替换表。"""

    #: (口语说法, 公文词)，按口语说法长度降序，保证长词先替换
    pairs: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.pairs)

    @classmethod
    def from_groups(cls, groups: Iterable[dict[str, object]]) -> "SynonymTable":
        pairs: list[tuple[str, str]] = []
        for group in groups:
            canonical = str(group.get("canonical") or "").strip()
            if not canonical:
                continue
            variants = group.get("variants") or []
            if isinstance(variants, str):
                variants = [variants]
            for variant in variants:  # type: ignore[union-attr]
                text = str(variant).strip()
                if text and text != canonical:
                    pairs.append((text, canonical))
        # 长词优先，避免短词先把长词吃掉
        pairs.sort(key=lambda item: len(item[0]), reverse=True)
        return cls(pairs=tuple(pairs))

    @classmethod
    def load(cls, path: str | Path) -> "SynonymTable":
        file = Path(path)
        if not file.is_file():
            return cls()
        try:
            data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return cls()
        groups = data.get("groups") if isinstance(data, dict) else None
        if not isinstance(groups, list):
            return cls()
        return cls.from_groups(groups)

    def normalize(self, text: str) -> str:
        """把口语说法替换成公文词；没有命中就原样返回。"""
        if not text or not self.pairs:
            return text
        result = text
        for variant, canonical in self.pairs:
            if variant in result:
                result = result.replace(variant, canonical)
        return result

    def matched(self, text: str) -> list[tuple[str, str]]:
        """返回文本里命中过的 (口语说法, 公文词)，供诊断用。"""
        return [(variant, canonical) for variant, canonical in self.pairs if variant in text]
