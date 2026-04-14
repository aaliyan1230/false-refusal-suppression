from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    split_group_targets: Optional[Mapping[str, Mapping[str, int]]] = None,
    seed: int = 0,
) -> Dict[str, List[PromptExample]]:
    grouped: MutableMapping[str, List[PromptExample]] = defaultdict(list)
    for example in examples:
        grouped[example.resolved_family_id].append(example)

    families = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(families)

    if split_group_targets is not None:
        return _make_grouped_splits_with_targets(families, split_group_targets)

    fractions = _normalize_split_fractions(split_fractions or DEFAULT_SPLIT_FRACTIONS)

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


def _make_grouped_splits_with_targets(
    families: Sequence[Tuple[str, List[PromptExample]]],
    split_group_targets: Mapping[str, Mapping[str, int]],
) -> Dict[str, List[PromptExample]]:
    available_by_group: Dict[str, int] = defaultdict(int)
    for _family_id, family_examples in families:
        for example in family_examples:
            available_by_group[example.group] += 1

    required_by_group: Dict[str, int] = defaultdict(int)
    for group_targets in split_group_targets.values():
        for group, count in group_targets.items():
            required_by_group[group] += count

    for group, required_count in required_by_group.items():
        available_count = available_by_group.get(group, 0)
        if required_count > available_count:
            raise ValueError(
                f'split_group_targets require {required_count} examples for group {group}, '
                f'but only {available_count} available examples exist'
            )
        if required_count != available_count:
            raise ValueError(
                f'split_group_targets must account for every example in group {group}: '
                f'required {required_count}, available {available_count}'
            )

    extra_groups = set(available_by_group) - set(required_by_group)
    if extra_groups:
        raise ValueError(
            'split_group_targets must account for every example group; missing groups: '
            + ', '.join(sorted(extra_groups))
        )

    remaining = {
        split_name: dict(group_targets)
        for split_name, group_targets in split_group_targets.items()
    }
    assignments = {split_name: [] for split_name in split_group_targets}
    ordered_families = sorted(families, key=lambda item: len(item[1]), reverse=True)

    if not _assign_family(0, ordered_families, remaining, assignments):
        raise ValueError('Could not satisfy split_group_targets while keeping prompt families together')

    return assignments


def _assign_family(
    index: int,
    families: Sequence[Tuple[str, List[PromptExample]]],
    remaining: MutableMapping[str, Dict[str, int]],
    assignments: MutableMapping[str, List[PromptExample]],
) -> bool:
    if index >= len(families):
        return all(count == 0 for group_counts in remaining.values() for count in group_counts.values())

    _family_id, family_examples = families[index]
    family_counts: Dict[str, int] = defaultdict(int)
    for example in family_examples:
        family_counts[example.group] += 1

    candidate_splits = sorted(
        remaining,
        key=lambda split_name: sum(remaining[split_name].get(group, 0) for group in family_counts),
        reverse=True,
    )

    for split_name in candidate_splits:
        if not _can_fit_family(family_counts, remaining[split_name]):
            continue

        for group, count in family_counts.items():
            remaining[split_name][group] = remaining[split_name].get(group, 0) - count
        assignments[split_name].extend(family_examples)

        if _assign_family(index + 1, families, remaining, assignments):
            return True

        del assignments[split_name][-len(family_examples):]
        for group, count in family_counts.items():
            remaining[split_name][group] = remaining[split_name].get(group, 0) + count

    return False


def _can_fit_family(family_counts: Mapping[str, int], split_remaining: Mapping[str, int]) -> bool:
    for group, count in family_counts.items():
        if split_remaining.get(group, 0) < count:
            return False
    return True
