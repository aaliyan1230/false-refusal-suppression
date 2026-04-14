#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frs.editing.apply_edit import (
    EditSpec,
    apply_direction_to_model,
    apply_directional_ablation,
    find_editable_modules,
    serialize_targets,
)
from frs.models.loader import ModelLoadConfig, load_model_and_tokenizer
from frs.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply a directional ablation to a matrix artifact or model weights.')
    parser.add_argument('--matrix')
    parser.add_argument('--direction')
    parser.add_argument('--model-id')
    parser.add_argument('--direction-artifact')
    parser.add_argument('--direction-layer')
    parser.add_argument('--module-type', action='append', dest='module_types')
    parser.add_argument('--layer', action='append', dest='layers', type=int)
    parser.add_argument('--save-model-dir')
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--norm-preserving', action='store_true')
    parser.add_argument('--axis', choices=('input', 'output'), default='output')
    args = parser.parse_args()

    if args.matrix and args.direction:
        matrix = read_json(args.matrix)
        direction = read_json(args.direction)
        edited = apply_directional_ablation(
            matrix,
            direction,
            EditSpec(strength=args.strength, norm_preserving=args.norm_preserving, axis=args.axis),
        )
        write_json(args.output, edited)
        print(args.output)
        return

    if not args.model_id or not args.direction_artifact:
        raise ValueError('Matrix mode requires --matrix and --direction; model mode requires --model-id and --direction-artifact')

    direction_payload = read_json(args.direction_artifact)
    selected_layer = args.direction_layer or _default_direction_layer(direction_payload)
    if not selected_layer:
        raise ValueError('Could not determine a direction layer from the artifact')

    direction_entry = direction_payload['directions'][selected_layer]
    direction = direction_entry['direction']
    applied_layers = args.layers or _default_applied_layers(selected_layer)
    target_module_types = args.module_types or ['attn_out']

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.model_id,
            load_in_4bit=args.load_in_4bit,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    targets = find_editable_modules(model, target_module_types=target_module_types, layers=applied_layers)
    if not targets:
        raise ValueError('No editable modules matched the requested layer/module filters')

    spec = EditSpec(strength=args.strength, norm_preserving=args.norm_preserving, axis=args.axis)
    apply_direction_to_model(model, direction, targets, spec)

    if args.save_model_dir:
        output_dir = Path(args.save_model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    artifact = {
        'model_id': args.model_id,
        'direction_layer': selected_layer,
        'source_separability_score': direction_entry.get('separability_score'),
        'applied_layers': sorted({target.layer_index for target in targets if target.layer_index is not None}),
        'module_types': target_module_types,
        'strength': args.strength,
        'norm_preserving': args.norm_preserving,
        'axis': args.axis,
        'targets': serialize_targets(targets),
        'saved_model_dir': args.save_model_dir,
    }
    write_json(args.output, artifact)
    print(args.output)



def _default_direction_layer(direction_payload: dict) -> str:
    ranked_layers = direction_payload.get('ranked_layers') or []
    if ranked_layers:
        return ranked_layers[0]['name']
    directions = direction_payload.get('directions') or {}
    return next(iter(directions), '')


def _default_applied_layers(direction_layer: str) -> list[int]:
    match = re.search(r'layer_(\d+)', direction_layer)
    if match is None:
        return []
    return [int(match.group(1))]


if __name__ == '__main__':
    main()
