from __future__ import annotations

from typing import Sequence


def exact_match_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError('predictions and references must have the same length')
    if not predictions:
        return 0.0
    matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    return matches / len(predictions)


def capability_retention(base_score: float, edited_score: float) -> float:
    if base_score <= 0:
        return 1.0 if edited_score >= base_score else 0.0
    ratio = edited_score / base_score
    return round(max(0.0, min(ratio, 1.0)), 12)
