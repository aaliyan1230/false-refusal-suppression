from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class EditCandidate:
    name: str
    false_refusal_rate: float
    true_refusal_rate: float
    capability_retention: float
    harmless_kl_penalty: float


def score_candidate(candidate: EditCandidate, w1: float = 1.0, w2: float = 1.0, w3: float = 1.0, w4: float = 1.0) -> float:
    return (
        w1 * (1.0 - candidate.false_refusal_rate)
        + w2 * candidate.true_refusal_rate
        + w3 * candidate.capability_retention
        - w4 * candidate.harmless_kl_penalty
    )


def rank_candidates(candidates: Iterable[EditCandidate]) -> List[EditCandidate]:
    return sorted(candidates, key=score_candidate, reverse=True)
