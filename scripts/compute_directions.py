#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frs.editing.directions import direction_from_contrast, rank_layers_by_separability, separability_score
from frs.editing.projection import orthogonalize
from frs.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute normalized directions from activation artifacts or raw vectors.')
    parser.add_argument('--group-a')
    parser.add_argument('--group-b')
    parser.add_argument('--activations')
    parser.add_argument('--source-group-a')
    parser.add_argument('--source-group-b')
    parser.add_argument('--reference-activations')
    parser.add_argument('--reference-group-a')
    parser.add_argument('--reference-group-b')
    parser.add_argument('--module-vectors', action='store_true')
    parser.add_argument('--top-k-layers', type=int)
    parser.add_argument('--orthogonalize', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if args.activations:
        if not args.source_group_a or not args.source_group_b:
            raise ValueError('--source-group-a and --source-group-b are required with --activations')

        payload = compute_from_artifact(
            activations=read_json(args.activations),
            source_group_a=args.source_group_a,
            source_group_b=args.source_group_b,
            vector_source='module_vectors' if args.module_vectors else 'layer_vectors',
            reference_activations=read_json(args.reference_activations) if args.reference_activations else None,
            reference_group_a=args.reference_group_a,
            reference_group_b=args.reference_group_b,
            orthogonalize_reference=args.orthogonalize,
            top_k_layers=args.top_k_layers,
        )
        write_json(args.output, payload)
        print(args.output)
        return

    if not args.group_a or not args.group_b:
        raise ValueError('Either activation artifact inputs or both --group-a and --group-b are required')

    group_a = read_json(args.group_a)
    group_b = read_json(args.group_b)
    payload = {
        'direction': direction_from_contrast(group_a, group_b),
        'separability_score': separability_score(group_a, group_b),
    }
    write_json(args.output, payload)
    print(args.output)


def compute_from_artifact(
    activations: dict,
    source_group_a: str,
    source_group_b: str,
    vector_source: str,
    reference_activations: dict | None,
    reference_group_a: str | None,
    reference_group_b: str | None,
    orthogonalize_reference: bool,
    top_k_layers: int | None,
) -> dict:
    grouped_ids = _group_example_ids(activations)
    vectors_by_name = activations.get(vector_source) or {}
    reference_grouped_ids = _group_example_ids(reference_activations) if reference_activations is not None else {}
    reference_vectors_by_name = (reference_activations or {}).get(vector_source) or {}

    directions = {}
    separability_by_name = {}
    for name, vectors_by_example in vectors_by_name.items():
        group_a = _select_vectors(vectors_by_example, grouped_ids.get(source_group_a, []))
        group_b = _select_vectors(vectors_by_example, grouped_ids.get(source_group_b, []))
        if not group_a or not group_b:
            continue

        direction = direction_from_contrast(group_a, group_b)
        reference_used = False
        if orthogonalize_reference:
            if reference_activations is None or not reference_group_a or not reference_group_b:
                raise ValueError('Reference artifact and groups are required when --orthogonalize is set')
            reference_group_a_vectors = _select_vectors(
                reference_vectors_by_name.get(name, {}),
                reference_grouped_ids.get(reference_group_a, []),
            )
            reference_group_b_vectors = _select_vectors(
                reference_vectors_by_name.get(name, {}),
                reference_grouped_ids.get(reference_group_b, []),
            )
            if reference_group_a_vectors and reference_group_b_vectors:
                direction = orthogonalize(
                    direction,
                    direction_from_contrast(reference_group_a_vectors, reference_group_b_vectors),
                )
                reference_used = True

        score = separability_score(group_a, group_b)
        separability_by_name[name] = score
        directions[name] = {
            'direction': direction,
            'separability_score': score,
            'num_group_a': len(group_a),
            'num_group_b': len(group_b),
            'orthogonalized': reference_used,
        }

    ranked = [
        {'name': name, 'score': score}
        for name, score in rank_layers_by_separability(separability_by_name)
    ]
    if top_k_layers is not None:
        ranked = ranked[:top_k_layers]

    return {
        'artifact_type': 'direction_collection',
        'vector_source': vector_source,
        'source_groups': {
            'group_a': source_group_a,
            'group_b': source_group_b,
        },
        'reference_groups': {
            'group_a': reference_group_a,
            'group_b': reference_group_b,
        }
        if orthogonalize_reference
        else None,
        'directions': directions,
        'ranked_layers': ranked,
    }


def _group_example_ids(activations: dict | None) -> dict:
    grouped_ids = {}
    for record in (activations or {}).get('records', []):
        grouped_ids.setdefault(record['group'], []).append(record['id'])
    return grouped_ids


def _select_vectors(vectors_by_example: dict, example_ids: list[str]) -> list[list[float]]:
    return [vectors_by_example[example_id] for example_id in example_ids if example_id in vectors_by_example]


if __name__ == '__main__':
    main()
