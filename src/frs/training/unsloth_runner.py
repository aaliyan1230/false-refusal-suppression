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
    if not dataset:
        raise ValueError('dataset must be non-empty')

    backend = _detect_backend()
    if backend == 'unsloth':
        return _run_unsloth_backend(config, dataset)
    if backend == 'transformers':
        return _run_transformers_backend(config, dataset)
    raise RuntimeError(
        'No supported training backend found. Install either unsloth or the transformers/peft/trl stack.'
    )


def _detect_backend() -> str:
    try:
        import unsloth  # noqa: F401
    except ImportError:
        pass
    else:
        return 'unsloth'

    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return 'none'
    return 'transformers'


def _run_unsloth_backend(config: QLoRAConfig, dataset: Sequence[dict]) -> dict:
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=list(config.target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias='none',
        use_gradient_checkpointing='unsloth',
    )

    train_dataset = Dataset.from_list(_format_dataset_text(dataset, tokenizer))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field='text',
        max_seq_length=config.max_seq_length,
        packing=False,
        args=TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=config.logging_steps,
            warmup_ratio=config.warmup_ratio,
            report_to='none',
            save_strategy='epoch',
            seed=config.seed,
        ),
    )
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return _build_result_payload(config, dataset, 'unsloth', train_result.metrics)


def _run_transformers_backend(config: QLoRAConfig, dataset: Sequence[dict]) -> dict:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {'device_map': 'auto'}
    if config.load_in_4bit:
        model_kwargs['load_in_4bit'] = True
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias='none',
    )
    model = get_peft_model(model, lora_config)

    train_dataset = Dataset.from_list(_format_dataset_text(dataset, tokenizer))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field='text',
        max_seq_length=config.max_seq_length,
        packing=False,
        args=TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=config.logging_steps,
            warmup_ratio=config.warmup_ratio,
            report_to='none',
            save_strategy='epoch',
            seed=config.seed,
        ),
    )
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return _build_result_payload(config, dataset, 'transformers', train_result.metrics)


def _format_dataset_text(dataset: Sequence[dict], tokenizer: object) -> List[dict]:
    formatted = []
    for record in dataset:
        messages = [
            {'role': 'user', 'content': record['prompt']},
            {'role': 'assistant', 'content': record['completion']},
        ]
        text = record['text']
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted.append({**record, 'text': text})
    return formatted


def _build_result_payload(config: QLoRAConfig, dataset: Sequence[dict], backend: str, metrics: dict) -> dict:
    return {
        'status': 'completed',
        'backend': backend,
        'model_id': config.model_id,
        'output_dir': config.output_dir,
        'num_examples': len(dataset),
        'lora_rank': config.lora_rank,
        'epochs': config.epochs,
        'metrics': dict(metrics),
    }
