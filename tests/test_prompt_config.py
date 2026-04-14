from __future__ import annotations

import json
from pathlib import Path

from frs.data.loaders import load_prompt_examples


def test_load_prompt_set_config_returns_exact_split_targets(tmp_path: Path):
    config_path = tmp_path / 'prompt_sets.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'groups:',
                '  benign_easy:',
                '    expected_behavior: answer',
                '  benign_borderline:',
                '    expected_behavior: answer',
                '  unsafe_true_refusal:',
                '    expected_behavior: refuse',
                'splits:',
                '  discovery:',
                '    benign_easy: 2',
                '    benign_borderline: 1',
                '  selection:',
                '    benign_easy: 1',
                '    unsafe_true_refusal: 1',
                '  holdout:',
                '    benign_borderline: 1',
                '    unsafe_true_refusal: 2',
                'rules:',
                '  grouped_family_split: true',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    from frs.data.config import load_prompt_set_config

    config = load_prompt_set_config(str(config_path))

    assert config.split_group_targets == {
        'discovery': {'benign_easy': 2, 'benign_borderline': 1},
        'selection': {'benign_easy': 1, 'unsafe_true_refusal': 1},
        'holdout': {'benign_borderline': 1, 'unsafe_true_refusal': 2},
    }
    assert config.expected_behavior_by_group['unsafe_true_refusal'] == 'refuse'
    assert config.rules['grouped_family_split'] is True


def test_create_split_manifests_uses_config_targets(tmp_path: Path):
    input_path = tmp_path / 'prompts.jsonl'
    config_path = tmp_path / 'prompt_sets.yaml'
    output_dir = tmp_path / 'splits'

    examples = [
        {
            'id': 'easy_1',
            'prompt': 'easy 1',
            'group': 'benign_easy',
            'topic': 'topic',
            'expected_behavior': 'answer',
            'source': 'human',
            'family_id': 'easy_family_1',
        },
        {
            'id': 'easy_2',
            'prompt': 'easy 2',
            'group': 'benign_easy',
            'topic': 'topic',
            'expected_behavior': 'answer',
            'source': 'human',
            'family_id': 'easy_family_2',
        },
        {
            'id': 'borderline_1',
            'prompt': 'borderline 1',
            'group': 'benign_borderline',
            'topic': 'topic',
            'expected_behavior': 'answer',
            'source': 'human',
            'family_id': 'borderline_family_1',
        },
        {
            'id': 'unsafe_1',
            'prompt': 'unsafe 1',
            'group': 'unsafe_true_refusal',
            'topic': 'topic',
            'expected_behavior': 'refuse',
            'source': 'human',
            'family_id': 'unsafe_family_1',
        },
    ]
    input_path.write_text(''.join(json.dumps(row) + '\n' for row in examples), encoding='utf-8')
    config_path.write_text(
        '\n'.join(
            [
                'groups:',
                '  benign_easy:',
                '    expected_behavior: answer',
                '  benign_borderline:',
                '    expected_behavior: answer',
                '  unsafe_true_refusal:',
                '    expected_behavior: refuse',
                'splits:',
                '  discovery:',
                '    benign_easy: 1',
                '    benign_borderline: 1',
                '  selection:',
                '    benign_easy: 1',
                '    unsafe_true_refusal: 1',
                '  holdout: {}',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    from frs.data.manifests import create_split_manifests

    summary = create_split_manifests(
        input_path=str(input_path),
        output_dir=str(output_dir),
        config_path=str(config_path),
        seed=7,
    )

    assert summary['discovery'] == {'benign_easy': 1, 'benign_borderline': 1}
    assert summary['selection'] == {'benign_easy': 1, 'unsafe_true_refusal': 1}
    assert summary['holdout'] == {}

    discovery_examples = load_prompt_examples(str(output_dir / 'discovery.jsonl'))
    selection_examples = load_prompt_examples(str(output_dir / 'selection.jsonl'))
    holdout_path = output_dir / 'holdout.jsonl'

    assert len(discovery_examples) == 2
    assert len(selection_examples) == 2
    assert holdout_path.exists()
    assert holdout_path.read_text(encoding='utf-8') == ''
