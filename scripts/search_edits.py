#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frs.data.loaders import load_prompt_examples
from frs.editing.apply_edit import (
    EditSpec,
    apply_direction_to_model,
    find_editable_modules,
    restore_module_weights,
    snapshot_module_weights,
)
from frs.editing.search import EditCandidate, rank_candidates
from frs.evaluation.capability import capability_retention
from frs.evaluation.drift import mean_kl_divergence
from frs.evaluation.refusal import false_refusal_rate, is_refusal, true_refusal_rate
from frs.models.generation import TextGenerationConfig, generate_text, next_token_distribution
from frs.models.loader import ModelLoadConfig, load_model_and_tokenizer
from frs.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Run live edit search or rank existing edit candidates.')
    parser.add_argument('--candidates')
    parser.add_argument('--model-id')
    parser.add_argument('--direction-artifact')
    parser.add_argument('--selection-split')
    parser.add_argument('--output', required=True)
    parser.add_argument('--top-k-layers', type=int, default=3)
    parser.add_argument('--strength', action='append', dest='strengths', type=float)
    parser.add_argument('--span-width', action='append', dest='span_widths', type=int)
    parser.add_argument('--module-type', action='append', dest='module_types')
    parser.add_argument('--include-norm-preserving', action='store_true')
    parser.add_argument('--axis', choices=('input', 'output', 'auto'), default='auto')
    parser.add_argument('--max-input-length', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--prompt-limit', type=int)
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--write-partial-results', action='store_true')
    args = parser.parse_args()

    if args.candidates and not args.model_id and not args.direction_artifact and not args.selection_split:
        raw_candidates = read_json(args.candidates)
        candidates = [EditCandidate(**payload) for payload in raw_candidates]
        ranked = [asdict(candidate) for candidate in rank_candidates(candidates)]
        write_json(args.output, ranked)
        print(args.output)
        return

    if not args.model_id or not args.direction_artifact or not args.selection_split:
        raise ValueError(
            'Live search requires --model-id, --direction-artifact, and --selection-split; ranking mode requires only --candidates'
        )

    direction_payload = read_json(args.direction_artifact)
    examples = load_prompt_examples(args.selection_split)
    if args.prompt_limit is not None:
        examples = examples[:args.prompt_limit]
    if not examples:
        raise ValueError('No examples selected for edit search')

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.model_id,
            load_in_4bit=args.load_in_4bit,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    generation_config = TextGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
        max_input_length=args.max_input_length,
    )

    base_metrics = evaluate_model(model, tokenizer, examples, generation_config)
    direction_layers = direction_payload.get('ranked_layers') or []
    direction_layers = direction_layers[: args.top_k_layers]
    if not direction_layers:
        direction_layers = [{'name': name, 'score': entry.get('separability_score', 0.0)} for name, entry in direction_payload.get('directions', {}).items()]

    all_targets = find_editable_modules(model, target_module_types=args.module_types or ['attn_out', 'mlp_down'])
    max_layer_index = max((target.layer_index for target in all_targets if target.layer_index is not None), default=0)

    strengths = args.strengths or [0.25, 0.5, 1.0]
    span_widths = args.span_widths or [1, 3]
    module_sets = build_module_sets(args.module_types or ['attn_out', 'mlp_down'])
    norm_options = [False, True] if args.include_norm_preserving else [False]

    search_plan = build_search_plan(
        direction_payload=direction_payload,
        top_k_layers=args.top_k_layers,
        strengths=strengths,
        span_widths=span_widths,
        module_sets=module_sets,
        norm_options=norm_options,
        max_layer_index=max_layer_index,
    )
    total_candidates = len(search_plan)
    print(f'Loaded {len(examples)} selection prompts')
    print(f'Base evaluation complete; evaluating {total_candidates} edit candidates')

    candidates = []
    for candidate_index, plan_item in enumerate(search_plan, start=1):
        source_layer = plan_item['source_layer']
        applied_layers = plan_item['applied_layers']
        module_types = plan_item['module_types']
        strength = plan_item['strength']
        norm_preserving = plan_item['norm_preserving']
        direction = direction_payload['directions'][source_layer]['direction']
        targets = find_editable_modules(model, target_module_types=module_types, layers=applied_layers)
        if not targets:
            print(
                f'Skipping candidate {candidate_index}/{total_candidates}: '
                f'{source_layer} layers={applied_layers} modules={module_types} (no matching targets)',
                flush=True,
            )
            continue

        snapshots = snapshot_module_weights(model, targets)
        started_at = time.perf_counter()
        print(
            f'[{candidate_index}/{total_candidates}] '
            f'source={source_layer} layers={applied_layers} modules={module_types} '
            f'strength={strength} norm_preserving={norm_preserving}',
            flush=True,
        )
        try:
            spec = EditSpec(strength=strength, norm_preserving=norm_preserving, axis=args.axis)
            apply_direction_to_model(model, direction, targets, spec)
            edited_metrics = evaluate_model(model, tokenizer, examples, generation_config)
            candidate = EditCandidate(
                name=build_candidate_name(source_layer, applied_layers, module_types, strength, norm_preserving),
                false_refusal_rate=false_refusal_rate(edited_metrics['responses_by_group'].get('benign_borderline', [])),
                true_refusal_rate=true_refusal_rate(edited_metrics['responses_by_group'].get('unsafe_true_refusal', [])),
                capability_retention=capability_retention(
                    base_metrics['capability_answer_rate'],
                    edited_metrics['capability_answer_rate'],
                ),
                harmless_kl_penalty=mean_kl_divergence(
                    base_metrics['capability_distributions'],
                    edited_metrics['capability_distributions'],
                ),
                source_layer=source_layer,
                applied_layers=tuple(applied_layers),
                strength=strength,
                target_modules=tuple(module_types),
                norm_preserving=norm_preserving,
                axis=args.axis,
                metadata={
                    'module_count': len(targets),
                    'separability_score': direction_payload['directions'][source_layer].get('separability_score'),
                    'capability_answer_rate': edited_metrics['capability_answer_rate'],
                },
            )
            candidates.append(candidate)
            duration_seconds = time.perf_counter() - started_at
            print(
                f'Completed [{candidate_index}/{total_candidates}] in {duration_seconds:.1f}s: '
                f'false_refusal={candidate.false_refusal_rate:.3f}, '
                f'true_refusal={candidate.true_refusal_rate:.3f}, '
                f'capability_retention={candidate.capability_retention:.3f}, '
                f'harmless_kl={candidate.harmless_kl_penalty:.6f}',
                flush=True,
            )
            if args.write_partial_results:
                ranked_partial = [asdict(item) for item in rank_candidates(candidates)]
                write_json(args.output, ranked_partial)
        finally:
            restore_module_weights(model, snapshots)

    ranked = [asdict(candidate) for candidate in rank_candidates(candidates)]
    write_json(args.output, ranked)
    print(args.output)


def build_search_plan(
    direction_payload: dict,
    top_k_layers: int,
    strengths: list[float],
    span_widths: list[int],
    module_sets: list[tuple[str, ...]],
    norm_options: list[bool],
    max_layer_index: int,
) -> list[dict]:
    direction_layers = direction_payload.get('ranked_layers') or []
    direction_layers = direction_layers[:top_k_layers]
    if not direction_layers:
        direction_layers = [
            {'name': name, 'score': entry.get('separability_score', 0.0)}
            for name, entry in direction_payload.get('directions', {}).items()
        ]

    plan = []
    for ranked_layer in direction_layers:
        source_layer = ranked_layer['name']
        source_index = parse_layer_index(source_layer)
        candidate_layer_spans = build_layer_spans(source_index, span_widths, max_layer_index)
        for applied_layers in candidate_layer_spans:
            for module_types in module_sets:
                for strength in strengths:
                    for norm_preserving in norm_options:
                        plan.append(
                            {
                                'source_layer': source_layer,
                                'applied_layers': applied_layers,
                                'module_types': module_types,
                                'strength': strength,
                                'norm_preserving': norm_preserving,
                            }
                        )
    return plan


def evaluate_model(model: object, tokenizer: object, examples: list, generation_config: TextGenerationConfig) -> dict:
    responses_by_group = {}
    capability_distributions = []
    capability_responses = []
    for example in examples:
        response = generate_text(model, tokenizer, example.prompt, generation_config)
        responses_by_group.setdefault(example.group, []).append(response)
        if example.group == 'capability_holdout':
            capability_responses.append(response)
            capability_distributions.append(
                next_token_distribution(
                    model,
                    tokenizer,
                    example.prompt,
                    max_input_length=generation_config.max_input_length,
                )
            )
    answer_count = sum(1 for response in capability_responses if not is_refusal(response))
    capability_answer_rate = answer_count / len(capability_responses) if capability_responses else 1.0
    return {
        'responses_by_group': responses_by_group,
        'capability_answer_rate': capability_answer_rate,
        'capability_distributions': capability_distributions,
    }


def build_module_sets(module_types: list[str]) -> list[tuple[str, ...]]:
    ordered = tuple(dict.fromkeys(module_types))
    module_sets = [(module_type,) for module_type in ordered]
    if len(ordered) > 1:
        module_sets.append(ordered)
    return module_sets


def build_layer_spans(source_index: int | None, span_widths: list[int], max_layer_index: int) -> list[list[int]]:
    if source_index is None:
        return [[]]

    spans = []
    for width in span_widths:
        radius = max((width - 1) // 2, 0)
        start = max(source_index - radius, 0)
        end = min(start + width - 1, max_layer_index)
        start = max(end - width + 1, 0)
        spans.append(list(range(start, end + 1)))
    deduped = []
    for span in spans:
        if span not in deduped:
            deduped.append(span)
    return deduped


def parse_layer_index(layer_name: str) -> int | None:
    match = re.search(r'layer_(\d+)', layer_name)
    if match is None:
        return None
    return int(match.group(1))


def build_candidate_name(
    source_layer: str,
    applied_layers: list[int],
    module_types: tuple[str, ...],
    strength: float,
    norm_preserving: bool,
) -> str:
    layer_label = 'none' if not applied_layers else '-'.join(str(layer) for layer in applied_layers)
    module_label = '+'.join(module_types)
    norm_label = 'norm' if norm_preserving else 'plain'
    return f'{source_layer}|layers={layer_label}|modules={module_label}|strength={strength}|{norm_label}'


if __name__ == '__main__':
    main()
