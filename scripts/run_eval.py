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

from frs.data.loaders import load_prompt_examples
from frs.editing.apply_edit import EditSpec, apply_direction_to_model, find_editable_modules
from frs.evaluation.capability import capability_retention
from frs.evaluation.drift import mean_kl_divergence
from frs.evaluation.metrics import CalibrationMetrics
from frs.evaluation.reports import render_text_report, write_json_report
from frs.evaluation.refusal import false_refusal_rate, is_refusal, true_refusal_rate
from frs.models.generation import TextGenerationConfig, generate_text, next_token_distribution
from frs.models.loader import ModelLoadConfig, load_model_and_tokenizer
from frs.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Write an evaluation report from scalar metrics or a live model evaluation.')
    parser.add_argument('--false-refusal-rate', type=float)
    parser.add_argument('--true-refusal-rate', type=float)
    parser.add_argument('--capability-retention', type=float)
    parser.add_argument('--harmless-kl-penalty', type=float)
    parser.add_argument('--model-id')
    parser.add_argument('--prompts')
    parser.add_argument('--direction-artifact')
    parser.add_argument('--direction-layer')
    parser.add_argument('--candidate-json')
    parser.add_argument('--candidate-index', type=int, default=0)
    parser.add_argument('--module-type', action='append', dest='module_types')
    parser.add_argument('--layer', action='append', dest='layers', type=int)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--norm-preserving', action='store_true')
    parser.add_argument('--axis', choices=('input', 'output'), default='output')
    parser.add_argument('--max-input-length', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--prompt-limit', type=int)
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if all(
        value is not None
        for value in (
            args.false_refusal_rate,
            args.true_refusal_rate,
            args.capability_retention,
            args.harmless_kl_penalty,
        )
    ):
        metrics = CalibrationMetrics(
            false_refusal_rate=args.false_refusal_rate,
            true_refusal_rate=args.true_refusal_rate,
            capability_retention=args.capability_retention,
            harmless_kl_penalty=args.harmless_kl_penalty,
        )
        payload = metrics.to_dict()
        write_json_report(args.output, payload)
        print(render_text_report(payload))
        return

    if not args.model_id or not args.prompts:
        raise ValueError('Live evaluation requires --model-id and --prompts, or provide all scalar metric inputs')

    examples = load_prompt_examples(args.prompts)
    if args.prompt_limit is not None:
        examples = examples[:args.prompt_limit]
    if not examples:
        raise ValueError('No prompts selected for evaluation')

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

    base_eval = evaluate_model(model, tokenizer, examples, generation_config)
    edit_descriptor = None
    if args.direction_artifact or args.candidate_json:
        direction, edit_descriptor = resolve_edit_configuration(args)
        applied_layers = edit_descriptor['applied_layers']
        module_types = edit_descriptor['module_types']
        targets = find_editable_modules(model, target_module_types=module_types, layers=applied_layers)
        if not targets:
            raise ValueError('No editable modules matched the requested edit configuration')
        apply_direction_to_model(
            model,
            direction,
            targets,
            EditSpec(
                strength=edit_descriptor['strength'],
                norm_preserving=edit_descriptor['norm_preserving'],
                axis=edit_descriptor['axis'],
            ),
        )

    edited_eval = evaluate_model(model, tokenizer, examples, generation_config)
    metrics = CalibrationMetrics(
        false_refusal_rate=false_refusal_rate(edited_eval['responses_by_group'].get('benign_borderline', [])),
        true_refusal_rate=true_refusal_rate(edited_eval['responses_by_group'].get('unsafe_true_refusal', [])),
        capability_retention=capability_retention(
            base_eval['capability_answer_rate'],
            edited_eval['capability_answer_rate'],
        ),
        harmless_kl_penalty=mean_kl_divergence(
            base_eval['capability_distributions'],
            edited_eval['capability_distributions'],
        ),
    )
    payload = {
        'mode': 'live',
        'model_id': args.model_id,
        'edit': edit_descriptor,
        'metrics': metrics.to_dict(),
        'group_metrics': {
            'benign_easy_refusal_rate': false_refusal_rate(edited_eval['responses_by_group'].get('benign_easy', [])),
            'benign_borderline_false_refusal_rate': metrics.false_refusal_rate,
            'unsafe_true_refusal_rate': metrics.true_refusal_rate,
            'capability_holdout_answer_rate': edited_eval['capability_answer_rate'],
        },
        'topic_breakdown': edited_eval['topic_breakdown'],
        'response_samples': edited_eval['response_samples'],
    }
    write_json_report(args.output, payload)
    print(render_text_report(payload['metrics']))


def evaluate_model(model: object, tokenizer: object, examples: list, generation_config: TextGenerationConfig) -> dict:
    responses_by_group = {}
    capability_distributions = []
    capability_responses = []
    topic_breakdown = {}
    response_samples = []
    for example in examples:
        response = generate_text(model, tokenizer, example.prompt, generation_config)
        responses_by_group.setdefault(example.group, []).append(response)
        topic_bucket = topic_breakdown.setdefault(example.topic, {'count': 0, 'refusals': 0})
        topic_bucket['count'] += 1
        topic_bucket['refusals'] += int(is_refusal(response))
        if len(response_samples) < 8:
            response_samples.append(
                {
                    'id': example.id,
                    'group': example.group,
                    'topic': example.topic,
                    'prompt': example.prompt,
                    'response': response,
                }
            )
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

    for topic, stats in topic_breakdown.items():
        stats['refusal_rate'] = stats['refusals'] / stats['count'] if stats['count'] else 0.0

    answer_count = sum(1 for response in capability_responses if not is_refusal(response))
    capability_answer_rate = answer_count / len(capability_responses) if capability_responses else 1.0
    return {
        'responses_by_group': responses_by_group,
        'capability_answer_rate': capability_answer_rate,
        'capability_distributions': capability_distributions,
        'topic_breakdown': topic_breakdown,
        'response_samples': response_samples,
    }


def resolve_edit_configuration(args: argparse.Namespace) -> tuple[list[float], dict]:
    if args.candidate_json:
        candidates = read_json(args.candidate_json)
        candidate = candidates[args.candidate_index]
        direction_payload = read_json(args.direction_artifact) if args.direction_artifact else None
        direction_layer = candidate['source_layer']
        direction = direction_payload['directions'][direction_layer]['direction'] if direction_payload else None
        if direction is None:
            raise ValueError('--direction-artifact is required when --candidate-json is used')
        return direction, {
            'source_layer': direction_layer,
            'applied_layers': list(candidate.get('applied_layers', [])),
            'module_types': list(candidate.get('target_modules', [])),
            'strength': float(candidate.get('strength', 1.0)),
            'norm_preserving': bool(candidate.get('norm_preserving', False)),
            'axis': candidate.get('axis', 'output'),
        }

    direction_payload = read_json(args.direction_artifact)
    direction_layer = args.direction_layer or _default_direction_layer(direction_payload)
    if not direction_layer:
        raise ValueError('Could not determine a direction layer from the artifact')
    direction = direction_payload['directions'][direction_layer]['direction']
    applied_layers = args.layers or _default_applied_layers(direction_layer)
    return direction, {
        'source_layer': direction_layer,
        'applied_layers': applied_layers,
        'module_types': args.module_types or ['attn_out'],
        'strength': args.strength,
        'norm_preserving': args.norm_preserving,
        'axis': args.axis,
    }


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
