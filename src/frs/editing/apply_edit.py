from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from frs.editing.norm_preserving import remove_direction_preserve_row_norm
from frs.editing.projection import project_vector


Matrix = List[List[float]]
INPUT_AXIS = 'input'
OUTPUT_AXIS = 'output'
AUTO_AXIS = 'auto'

ATTN_OUT_PATTERNS = (
    '.self_attn.o_proj',
    '.self_attn.out_proj',
    '.attention.o_proj',
    '.attention.out_proj',
    '.attn.out_proj',
    '.attn.c_proj',
    '.attention.wo',
)

MLP_DOWN_PATTERNS = (
    '.mlp.down_proj',
    '.mlp.fc2',
    '.mlp.c_proj',
    '.feed_forward.w2',
    '.feed_forward.down_proj',
)


@dataclass(frozen=True)
class EditSpec:
    strength: float = 1.0
    norm_preserving: bool = False
    axis: str = AUTO_AXIS

    def validate(self) -> None:
        if self.axis not in {INPUT_AXIS, OUTPUT_AXIS, AUTO_AXIS}:
            raise ValueError(f'Unsupported edit axis: {self.axis}')


@dataclass(frozen=True)
class ModelEditTarget:
    module_name: str
    module_type: str
    layer_index: Optional[int]
    input_dim: int
    output_dim: int


def apply_directional_ablation(matrix: Sequence[Sequence[float]], direction: Sequence[float], spec: EditSpec = EditSpec()) -> Matrix:
    spec.validate()
    if spec.axis == OUTPUT_AXIS:
        transposed = _transpose_matrix(matrix)
        edited = _apply_row_space_ablation(
            transposed,
            direction,
            EditSpec(strength=spec.strength, norm_preserving=spec.norm_preserving, axis=INPUT_AXIS),
        )
        return _transpose_matrix(edited)

    return _apply_row_space_ablation(matrix, direction, spec)


def apply_directional_ablation_tensor(matrix: object, direction: Sequence[float], spec: EditSpec = EditSpec()) -> object:
    torch = _require_torch()
    spec.validate()

    if getattr(matrix, 'ndim', None) != 2:
        raise ValueError('matrix must be rank-2')

    direction_tensor = torch.as_tensor(direction, dtype=matrix.dtype, device=matrix.device).flatten()
    norm = torch.linalg.vector_norm(direction_tensor)
    if float(norm) <= 1e-12:
        raise ValueError('direction must have non-zero norm')
    direction_tensor = direction_tensor / norm

    # Auto-resolve axis: pick whichever dimension matches the direction
    resolved_axis = spec.axis
    if resolved_axis == AUTO_AXIS:
        d = direction_tensor.numel()
        input_match = (matrix.shape[1] == d)
        output_match = (matrix.shape[0] == d)
        if input_match and not output_match:
            resolved_axis = INPUT_AXIS
        elif output_match and not input_match:
            resolved_axis = OUTPUT_AXIS
        elif input_match and output_match:
            resolved_axis = INPUT_AXIS  # square matrix: default to input
        else:
            raise ValueError(
                f'Auto axis: direction dim {d} matches neither input ({matrix.shape[1]}) nor output ({matrix.shape[0]})'
            )

    if resolved_axis == INPUT_AXIS:
        if matrix.shape[1] != direction_tensor.numel():
            raise ValueError(
                f'Input-axis edit requires direction width {matrix.shape[1]}, got {direction_tensor.numel()}'
            )
        projected = torch.outer(matrix @ direction_tensor, direction_tensor)
        edited = matrix - (spec.strength * projected)
        if spec.norm_preserving:
            original_norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
            edited_norms = torch.linalg.vector_norm(edited, dim=1, keepdim=True).clamp_min(1e-12)
            edited = edited * (original_norms / edited_norms)
        return edited

    if matrix.shape[0] != direction_tensor.numel():
        raise ValueError(
            f'Output-axis edit requires direction width {matrix.shape[0]}, got {direction_tensor.numel()}'
        )
    projected = torch.outer(direction_tensor, direction_tensor @ matrix)
    edited = matrix - (spec.strength * projected)
    if spec.norm_preserving:
        original_norms = torch.linalg.vector_norm(matrix, dim=0, keepdim=True)
        edited_norms = torch.linalg.vector_norm(edited, dim=0, keepdim=True).clamp_min(1e-12)
        edited = edited * (original_norms / edited_norms)
    return edited


def find_editable_modules(
    model: object,
    target_module_types: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
) -> List[ModelEditTarget]:
    selected_types = set(target_module_types or ('attn_out', 'mlp_down'))
    selected_layers = set(layers) if layers is not None else None
    targets: List[ModelEditTarget] = []
    for module_name, module in model.named_modules():
        weight = getattr(module, 'weight', None)
        if getattr(weight, 'ndim', None) != 2:
            continue
        module_type = infer_module_type(module_name)
        if module_type is None or module_type not in selected_types:
            continue
        layer_index = extract_layer_index(module_name)
        if selected_layers is not None and layer_index not in selected_layers:
            continue
        targets.append(
            ModelEditTarget(
                module_name=module_name,
                module_type=module_type,
                layer_index=layer_index,
                input_dim=int(weight.shape[1]),
                output_dim=int(weight.shape[0]),
            )
        )
    return targets


def snapshot_module_weights(model: object, targets: Sequence[ModelEditTarget]) -> Dict[str, object]:
    return {
        target.module_name: dict(model.named_modules())[target.module_name].weight.detach().clone()
        for target in targets
    }


def restore_module_weights(model: object, snapshots: Dict[str, object]) -> None:
    modules = dict(model.named_modules())
    for module_name, weight in snapshots.items():
        modules[module_name].weight.data.copy_(weight)


def apply_direction_to_model(
    model: object,
    direction: Sequence[float],
    targets: Sequence[ModelEditTarget],
    spec: EditSpec = EditSpec(),
) -> List[ModelEditTarget]:
    modules = dict(model.named_modules())
    applied: List[ModelEditTarget] = []
    for target in targets:
        module = modules[target.module_name]
        weight = module.weight.data
        # Skip modules where neither dimension matches the direction
        d = len(direction)
        if spec.axis == AUTO_AXIS and weight.shape[0] != d and weight.shape[1] != d:
            continue
        edited_weight = apply_directional_ablation_tensor(weight, direction, spec)
        module.weight.data.copy_(edited_weight)
        applied.append(target)
    return applied


def infer_module_type(module_name: str) -> Optional[str]:
    normalized = module_name.lower()
    if any(pattern in normalized for pattern in ATTN_OUT_PATTERNS):
        return 'attn_out'
    if any(pattern in normalized for pattern in MLP_DOWN_PATTERNS):
        return 'mlp_down'
    return None


def extract_layer_index(module_name: str) -> Optional[int]:
    match = re.search(r'(?:layers|layer|h|blocks|block)\.(\d+)', module_name)
    if match is None:
        return None
    return int(match.group(1))


def serialize_targets(targets: Iterable[ModelEditTarget]) -> List[dict]:
    return [
        {
            'module_name': target.module_name,
            'module_type': target.module_type,
            'layer_index': target.layer_index,
            'input_dim': target.input_dim,
            'output_dim': target.output_dim,
        }
        for target in targets
    ]


def _apply_row_space_ablation(matrix: Sequence[Sequence[float]], direction: Sequence[float], spec: EditSpec) -> Matrix:
    if spec.norm_preserving:
        return remove_direction_preserve_row_norm(matrix, direction, strength=spec.strength)

    updated = []
    for row in matrix:
        projected = project_vector(row, direction)
        updated.append([float(value) - (spec.strength * float(component)) for value, component in zip(row, projected)])
    return updated


def _transpose_matrix(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def _require_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required for model-space edits') from exc
    return torch
