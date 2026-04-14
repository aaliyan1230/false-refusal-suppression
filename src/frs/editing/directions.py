from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Vector = List[float]


def mean_vector(vectors: Sequence[Sequence[float]]) -> Vector:
    if not vectors:
        raise ValueError('vectors must be non-empty')
    width = len(vectors[0])
    if width == 0:
        raise ValueError('vectors must have non-zero width')
    totals = [0.0] * width
    for vector in vectors:
        if len(vector) != width:
            raise ValueError('all vectors must share the same width')
        for index, value in enumerate(vector):
            totals[index] += float(value)
    return [value / len(vectors) for value in totals]


def subtract_vectors(left: Sequence[float], right: Sequence[float]) -> Vector:
    if len(left) != len(right):
        raise ValueError('vectors must have identical length')
    return [float(a) - float(b) for a, b in zip(left, right)]


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def normalize_vector(vector: Sequence[float], eps: float = 1e-12) -> Vector:
    norm = l2_norm(vector)
    if norm <= eps:
        raise ValueError('cannot normalize a near-zero vector')
    return [float(value) / norm for value in vector]


def difference_of_means(group_a: Sequence[Sequence[float]], group_b: Sequence[Sequence[float]]) -> Vector:
    return subtract_vectors(mean_vector(group_a), mean_vector(group_b))


def direction_from_contrast(group_a: Sequence[Sequence[float]], group_b: Sequence[Sequence[float]]) -> Vector:
    return normalize_vector(difference_of_means(group_a, group_b))


def cosine_similarity(left: Sequence[float], right: Sequence[float], eps: float = 1e-12) -> float:
    denominator = max(l2_norm(left) * l2_norm(right), eps)
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    return dot / denominator


def separability_score(group_a: Sequence[Sequence[float]], group_b: Sequence[Sequence[float]]) -> float:
    delta = difference_of_means(group_a, group_b)
    distance = l2_norm(delta)
    return distance / (1.0 + _mean_centered_radius(group_a) + _mean_centered_radius(group_b))


def _mean_centered_radius(group: Sequence[Sequence[float]]) -> float:
    center = mean_vector(group)
    radii = [l2_norm(subtract_vectors(row, center)) for row in group]
    return sum(radii) / max(len(radii), 1)


def rank_layers_by_separability(layer_scores: Mapping[str, float]) -> List[Tuple[str, float]]:
    return sorted(layer_scores.items(), key=lambda item: item[1], reverse=True)
