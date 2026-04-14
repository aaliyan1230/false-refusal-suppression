from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from frs.data.schemas import PromptExample
from frs.training.qlora import QLoRAConfig, build_calibration_records


@dataclass(frozen=True)
class UnslothAvailability:
    supported_models: Sequence[str]

    def supports(self, model_id: str) -> bool:
        return model_id in set(self.supported_models)


def choose_repair_model(primary_model_id: str, availability: UnslothAvailability, fallback_model_id: str) -> str:
    return primary_model_id if availability.supports(primary_model_id) else fallback_model_id


def build_unsloth_dataset(examples: Iterable[PromptExample]) -> List[dict]:
    return build_calibration_records(examples)


def run_unsloth_training(config: QLoRAConfig, dataset: Sequence[dict]) -> dict:
    return {
        'status': 'scaffold_only',
        'model_id': config.model_id,
        'num_examples': len(dataset),
        'lora_rank': config.lora_rank,
    }
