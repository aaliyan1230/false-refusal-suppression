#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frs.data.loaders import load_prompt_examples
from frs.training.qlora import QLoRAConfig
from frs.training.unsloth_runner import (
    UnslothAvailability,
    build_unsloth_dataset,
    choose_repair_model,
    run_unsloth_training,
)
from frs.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Run QLoRA repair training with Unsloth or a transformers fallback.')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fallback-model-id')
    parser.add_argument('--supported-unsloth-model', action='append', dest='supported_unsloth_models', default=[])
    parser.add_argument('--lora-rank', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--per-device-train-batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-4bit', action='store_true')
    args = parser.parse_args()

    examples = load_prompt_examples(args.prompts)
    selected_model_id = args.model_id
    if args.fallback_model_id:
        selected_model_id = choose_repair_model(
            args.model_id,
            UnslothAvailability(tuple(args.supported_unsloth_models)),
            args.fallback_model_id,
        )

    config = QLoRAConfig(
        model_id=selected_model_id,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        load_in_4bit=not args.no_4bit,
    )
    dataset = build_unsloth_dataset(examples)
    result = run_unsloth_training(config, dataset)
    result['requested_model_id'] = args.model_id
    write_json(args.output, result)
    print(args.output)


if __name__ == '__main__':
    main()
