from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from frs.data.schemas import PromptExample

DEFAULT_SPLIT_FRACTIONS = {
    'discovery': 0.5,
    'selection': 0.2,
    'holdout': 0.3,
}


def _normalize_split_fractions(split_fractions: Mapping[str, float]) -> Dict[str, float]:
    total = float(sum(split_fractions.values()))
    if total <= 0:
        raise ValueError('split_fractions must sum to a positive value')
    return {name: value / total for name, value in split_fractions.items()}


def make_grouped_splits(
    examples: Iterable[PromptExample],
    split_fractions: Optional[Mapping[str, float]] = None,
    seed: int = 0,
) -> Dict[str, List[PromptExample]]:
    fractions = _normalize_split_fractions(split_fractions or DEFAULT_SPLIT_FRACTIONS)
    grouped: MutableMapping[str, List[PromptExample]] = defaultdict(list)
    for example in examples:
        grouped[example.resolved_family_id].append(example)

    families = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(families)

    split_names = list(fractions.keys())
    assignments = {name: [] for name in split_names}
    targets = {name: fractions[name] * sum(len(items) for _, items in families) for name in split_names}

    for family_id, family_examples in families:
        current = min(split_names, key=lambda name: len(assignments[name]) - targets[name])
        assignments[current].extend(family_examples)

    return assignments


def summarize_splits(splits: Mapping[str, Iterable[PromptExample]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split_name, split_examples in splits.items():
        group_counts: Dict[str, int] = {}
        for example in split_examples:
            group_counts[example.group] = group_counts.get(example.group, 0) + 1
        summary[split_name] = group_counts
    return summary
