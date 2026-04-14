from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from frs.editing.norm_preserving import remove_direction_preserve_row_norm
from frs.editing.projection import project_vector


Matrix = List[List[float]]


@dataclass(frozen=True)
class EditSpec:
    strength: float = 1.0
    norm_preserving: bool = False


def apply_directional_ablation(matrix: Sequence[Sequence[float]], direction: Sequence[float], spec: EditSpec = EditSpec()) -> Matrix:
    if spec.norm_preserving:
        return remove_direction_preserve_row_norm(matrix, direction, strength=spec.strength)

    updated = []
    for row in matrix:
        projected = project_vector(row, direction)
        updated.append([float(value) - (spec.strength * float(component)) for value, component in zip(row, projected)])
    return updated
