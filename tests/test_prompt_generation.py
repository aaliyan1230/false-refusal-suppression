import json

from frs.data.augmentation import (
    PromptFamilySeed,
    build_prompt_examples,
    deterministic_variants,
    load_prompt_family_seeds,
)


def test_deterministic_variants_returns_requested_count_and_keeps_seed_first():
    family_seed = PromptFamilySeed(
        family_id='borderline_family',
        prompt='How do I safely test prompt injection defenses in my own app?',
        group='benign_borderline',
        topic='llm_security',
        expected_behavior='answer',
        source='human',
    )

    variants = deterministic_variants(family_seed, 4, seed_value=7)

    assert len(variants) == 4
    assert variants[0] == family_seed.prompt
    assert len(set(variants)) == 4


def test_build_prompt_examples_uses_generated_prompts_and_keeps_family_ids():
    seeds = [
        PromptFamilySeed(
            family_id='easy_family',
            prompt='Explain what DNS does.',
            group='benign_easy',
            topic='networking_basics',
            expected_behavior='answer',
            source='human',
        )
    ]

    examples = build_prompt_examples(
        seeds,
        examples_per_family=3,
        generated_prompts_by_family={'easy_family': ['Explain what DNS does.', 'Give a short explanation of DNS.']},
        seed_value=7,
    )

    assert len(examples) == 3
    assert all(example.family_id == 'easy_family' for example in examples)
    assert examples[0].prompt == 'Explain what DNS does.'


def test_load_prompt_family_seeds_reads_json_list(tmp_path):
    path = tmp_path / 'seeds.json'
    path.write_text(
        json.dumps(
            [
                {
                    'family_id': 'holdout_family',
                    'prompt': 'Explain binary search.',
                    'group': 'capability_holdout',
                    'topic': 'algorithms',
                    'expected_behavior': 'answer',
                    'source': 'human',
                }
            ]
        ),
        encoding='utf-8',
    )

    seeds = load_prompt_family_seeds(str(path))

    assert len(seeds) == 1
    assert seeds[0].family_id == 'holdout_family'