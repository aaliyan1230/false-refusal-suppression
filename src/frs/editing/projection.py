from __future__ import annotations

from typing import List, Sequence

from frs.editing.directions import normalize_vector


Vector = List[float]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError('vectors must have identical length')
    return sum(float(a) * float(b) for a, b in zip(left, right))


def project_vector(vector: Sequence[float], onto: Sequence[float]) -> Vector:
    basis = normalize_vector(onto)
    scale = dot(vector, basis)
    return [scale * value for value in basis]


def orthogonalize(candidate: Sequence[float], reference: Sequence[float]) -> Vector:
    projection = project_vector(candidate, reference)
    return [float(value) - float(component) for value, component in zip(candidate, projection)]
