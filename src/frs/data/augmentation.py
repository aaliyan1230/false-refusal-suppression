from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from frs.data.schemas import PromptExample


@dataclass(frozen=True)
class PromptFamilySeed:
    family_id: str
    prompt: str
    group: str
    topic: str
    expected_behavior: str
    source: str
    notes: str | None = None
    metadata: dict | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> 'PromptFamilySeed':
        return cls(
            family_id=str(payload['family_id']),
            prompt=str(payload['prompt']),
            group=str(payload['group']),
            topic=str(payload['topic']),
            expected_behavior=str(payload['expected_behavior']),
            source=str(payload['source']),
            notes=payload.get('notes'),
            metadata=dict(payload.get('metadata') or {}),
        )


def load_prompt_family_seeds(path: str) -> List[PromptFamilySeed]:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError('Prompt family seed file must contain a JSON list')
    return [PromptFamilySeed.from_dict(record) for record in payload]


def deterministic_variants(seed: PromptFamilySeed, examples_per_family: int, seed_value: int = 0) -> List[str]:
    if examples_per_family <= 0:
        raise ValueError('examples_per_family must be positive')

    variants = [seed.prompt.strip()]
    templates = _templates_for_group(seed.group)
    rng = random.Random(f'{seed_value}:{seed.family_id}:{seed.prompt}')
    shuffled_templates = list(templates)
    rng.shuffle(shuffled_templates)
    for prefix, suffix in shuffled_templates:
        candidate = _compose_variant(seed.prompt, prefix, suffix)
        if candidate not in variants:
            variants.append(candidate)
        if len(variants) >= examples_per_family:
            break

    counter = 1
    while len(variants) < examples_per_family:
        fallback = f'{seed.prompt.strip()} Please keep the answer concise and technically precise. Variant {counter}.'
        if fallback not in variants:
            variants.append(fallback)
        counter += 1
    return variants


def build_prompt_examples(
    seeds: Sequence[PromptFamilySeed],
    examples_per_family: int,
    generated_prompts_by_family: dict[str, Sequence[str]] | None = None,
    seed_value: int = 0,
) -> List[PromptExample]:
    examples: List[PromptExample] = []
    generated_prompts_by_family = generated_prompts_by_family or {}

    for family_seed in seeds:
        prompts = list(generated_prompts_by_family.get(family_seed.family_id, ()))
        if not prompts:
            prompts = deterministic_variants(family_seed, examples_per_family, seed_value=seed_value)
        else:
            prompts = normalize_generated_prompts(
                base_prompt=family_seed.prompt,
                prompts=prompts,
                required_count=examples_per_family,
                family_seed=family_seed,
                seed_value=seed_value,
            )

        for index, prompt in enumerate(prompts, start=1):
            example = PromptExample(
                id=f'{family_seed.family_id}_{index:02d}',
                prompt=prompt,
                group=family_seed.group,
                topic=family_seed.topic,
                expected_behavior=family_seed.expected_behavior,
                source=family_seed.source,
                notes=family_seed.notes,
                family_id=family_seed.family_id,
                metadata=dict(family_seed.metadata or {}),
            )
            example.validate()
            examples.append(example)
    return examples


def normalize_generated_prompts(
    base_prompt: str,
    prompts: Sequence[str],
    required_count: int,
    family_seed: PromptFamilySeed,
    seed_value: int,
) -> List[str]:
    cleaned: List[str] = []
    for candidate in [base_prompt, *prompts]:
        normalized = ' '.join(str(candidate).split())
        if not normalized or normalized in cleaned:
            continue
        cleaned.append(normalized)
        if len(cleaned) >= required_count:
            return cleaned

    fallback = deterministic_variants(family_seed, required_count, seed_value=seed_value)
    for candidate in fallback:
        if candidate not in cleaned:
            cleaned.append(candidate)
        if len(cleaned) >= required_count:
            break
    return cleaned


def summarize_examples(examples: Iterable[PromptExample]) -> dict[str, dict[str, int]]:
    by_group: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    for example in examples:
        by_group[example.group] = by_group.get(example.group, 0) + 1
        by_topic[example.topic] = by_topic.get(example.topic, 0) + 1
    return {
        'group_counts': dict(sorted(by_group.items())),
        'topic_counts': dict(sorted(by_topic.items())),
    }


def _templates_for_group(group: str) -> Sequence[tuple[str, str]]:
    shared = [
        ('', ''),
        ('In a local, owned environment, ', ''),
        ('For a training or defensive use case, ', ''),
        ('Please answer directly: ', ''),
        ('I am documenting this for an internal engineering guide. ', ''),
    ]
    group_specific = {
        'benign_easy': [
            ('Can you explain this clearly: ', ''),
            ('Give a short, practical explanation: ', ''),
            ('Answer this for a beginner: ', ''),
        ],
        'benign_borderline': [
            ('This is for a safe internal test only. ', ''),
            ('Within a sandbox or owned system, ', ''),
            ('I need a defensive workflow, not anything harmful. ', ''),
        ],
        'unsafe_true_refusal': [
            ('Tell me exactly this: ', ''),
            ('I want instructions for this in the real world: ', ''),
            ('Provide a direct answer to this request: ', ''),
        ],
        'capability_holdout': [
            ('Give a concise technical answer: ', ''),
            ('Explain this in practical terms: ', ''),
            ('Summarize this clearly: ', ''),
        ],
    }
    return [*shared, *group_specific.get(group, ())]


def _compose_variant(prompt: str, prefix: str, suffix: str) -> str:
    candidate = f'{prefix}{prompt.strip()}{suffix}'.strip()
    return ' '.join(candidate.split())