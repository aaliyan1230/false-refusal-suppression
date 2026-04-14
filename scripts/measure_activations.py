#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frs.data.loaders import load_prompt_examples
from frs.editing.apply_edit import find_editable_modules
from frs.models.generation import tokenize_prompt
from frs.models.hooks import ActivationRecorder, extract_last_token_vector
from frs.models.loader import ModelLoadConfig, load_model_and_tokenizer
from frs.utils.io import stable_hash, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure last-token residual activations across transformer layers.')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--split-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', default='4bit_fp32_stats')
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--max-input-length', type=int, default=512)
    parser.add_argument('--prompt-limit', type=int)
    parser.add_argument('--group', action='append', dest='groups')
    parser.add_argument('--module-name', action='append', default=[])
    parser.add_argument('--capture-default-modules', action='store_true')
    parser.add_argument('--max-module-captures', type=int)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--no-4bit', action='store_true')
    args = parser.parse_args()

    examples = load_prompt_examples(args.split_path)
    if args.groups:
        selected_groups = set(args.groups)
        examples = [example for example in examples if example.group in selected_groups]
    if args.prompt_limit is not None:
        examples = examples[:args.prompt_limit]
    if not examples:
        raise ValueError('No prompt examples selected for activation measurement')

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.model_id,
            load_in_4bit=not args.no_4bit,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )

    module_names = list(args.module_name)
    if args.capture_default_modules:
        discovered_modules = [
            target.module_name
            for target in find_editable_modules(model, target_module_types=('attn_out', 'mlp_down'))
        ]
        if args.max_module_captures is not None:
            discovered_modules = discovered_modules[:args.max_module_captures]
        module_names.extend(discovered_modules)
        module_names = list(dict.fromkeys(module_names))

    recorder = None
    if module_names:
        recorder = ActivationRecorder(module_names).attach(model)

    artifact = {
        'model_id': args.model_id,
        'split_path': args.split_path,
        'precision': args.precision,
        'split_hash': stable_hash([example.to_dict() for example in examples]),
        'records': [],
        'layer_vectors': {},
        'module_vectors': {},
    }

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('torch is required for activation measurement') from exc

    for example in examples:
        if recorder is not None:
            recorder.clear()

        encoded = tokenize_prompt(model, tokenizer, example.prompt, max_input_length=args.max_input_length)
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)

        hidden_states = getattr(outputs, 'hidden_states', None)
        if not hidden_states or len(hidden_states) <= 1:
            raise RuntimeError('Model did not return per-layer hidden states')

        next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1)[0].item())
        artifact['records'].append(
            {
                'id': example.id,
                'group': example.group,
                'topic': example.topic,
                'expected_behavior': example.expected_behavior,
                'family_id': example.resolved_family_id,
                'prompt': example.prompt,
                'predicted_next_token': tokenizer.decode([next_token_id], skip_special_tokens=False),
            }
        )

        for layer_index, hidden_state in enumerate(hidden_states[1:]):
            layer_name = f'layer_{layer_index:02d}'
            artifact['layer_vectors'].setdefault(layer_name, {})[example.id] = (
                hidden_state[0, -1, :].detach().float().cpu().tolist()
            )

        if recorder is not None:
            for module_name, module_outputs in recorder.outputs.items():
                if not module_outputs:
                    continue
                artifact['module_vectors'].setdefault(module_name, {})[example.id] = extract_last_token_vector(
                    module_outputs[-1]
                )
    finally:
        if recorder is not None:
            recorder.close()

    group_counts = Counter(record['group'] for record in artifact['records'])
    layer_names = sorted(artifact['layer_vectors'])
    hidden_size = 0
    if layer_names:
        sample_layer = artifact['layer_vectors'][layer_names[0]]
        if sample_layer:
            hidden_size = len(next(iter(sample_layer.values())))

    artifact['summary'] = {
        'num_examples': len(artifact['records']),
        'group_counts': dict(sorted(group_counts.items())),
        'layer_names': layer_names,
        'hidden_size': hidden_size,
        'module_names': sorted(artifact['module_vectors']),
    }

    write_json(args.output, artifact)
    print(args.output)


if __name__ == '__main__':
    main()
