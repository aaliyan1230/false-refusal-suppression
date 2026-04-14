from __future__ import annotations

from typing import List, Sequence

from frs.editing.directions import l2_norm
from frs.editing.projection import project_vector


Matrix = List[List[float]]


def remove_direction_preserve_row_norm(matrix: Sequence[Sequence[float]], direction: Sequence[float], strength: float = 1.0) -> Matrix:
    updated = []
    for row in matrix:
        original_norm = l2_norm(row)
        projected = project_vector(row, direction)
        candidate = [float(value) - (strength * float(component)) for value, component in zip(row, projected)]
        candidate_norm = l2_norm(candidate)
        if original_norm == 0 or candidate_norm == 0:
            updated.append(list(candidate))
            continue
        scale = original_norm / candidate_norm
        updated.append([scale * value for value in candidate])
    return updated
