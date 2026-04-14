from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from frs.data.schemas import PromptExample


@dataclass(frozen=True)
class QLoRAConfig:
    model_id: str
    output_dir: str = 'artifacts/repair/latest'
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 1024
    epochs: int = 2
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    warmup_ratio: float = 0.03
    seed: int = 0
    load_in_4bit: bool = True
    target_modules: Sequence[str] = (
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
    )


def build_calibration_records(examples: Iterable[PromptExample]) -> List[dict]:
    records = []
    for example in examples:
        completion = build_target_completion(example)
        records.append({
            'id': example.id,
            'prompt': example.prompt,
            'completion': completion,
            'target_behavior': 'REFUSE' if example.expected_behavior == 'refuse' else 'ANSWER',
            'group': example.group,
            'topic': example.topic,
            'text': format_sft_record(example.prompt, completion),
        })
    return records


def build_target_completion(example: PromptExample) -> str:
    custom_target = example.metadata.get('target_response')
    if isinstance(custom_target, str) and custom_target.strip():
        return custom_target.strip()

    if example.expected_behavior == 'refuse':
        return "I can't assist with harmful, illegal, or non-consensual activity."

    topic_templates = {
        'shell_process_management': 'Use standard operating system tools to inspect the process, terminate it safely, and verify that the system is healthy afterward.',
        'defensive_security_testing': 'Describe a safe reproduction workflow that stays inside a local or owned environment, includes setup constraints, and ends with validation of the fix.',
        'llm_security': 'Outline a sandboxed defensive evaluation workflow, including synthetic test cases, logging, and criteria for deciding whether a defense worked.',
    }
    return topic_templates.get(
        example.topic,
        'Provide a direct, technically accurate answer with a short explanation and a simple example when useful.',
    )


def format_sft_record(prompt: str, completion: str) -> str:
    return f'User: {prompt}\nAssistant: {completion}'
