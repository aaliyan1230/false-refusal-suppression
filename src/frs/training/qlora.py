from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from frs.data.schemas import PromptExample


@dataclass(frozen=True)
class QLoRAConfig:
    model_id: str
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 1024
    epochs: int = 2
    learning_rate: float = 2e-4


def build_calibration_records(examples: Iterable[PromptExample]) -> List[dict]:
    records = []
    for example in examples:
        target = 'REFUSE' if example.expected_behavior == 'refuse' else 'ANSWER'
        records.append({
            'id': example.id,
            'prompt': example.prompt,
            'target_behavior': target,
            'group': example.group,
        })
    return records
