from __future__ import annotations

from typing import Iterable, List, Mapping

from frs.data.schemas import PromptExample


def normalize_prompt_record(record: Mapping[str, object]) -> PromptExample:
    return PromptExample.from_dict(dict(record))


def filter_by_group(examples: Iterable[PromptExample], group: str) -> List[PromptExample]:
    return [example for example in examples if example.group == group]


def group_counts(examples: Iterable[PromptExample]) -> dict:
    counts = {}
    for example in examples:
        counts[example.group] = counts.get(example.group, 0) + 1
    return counts
