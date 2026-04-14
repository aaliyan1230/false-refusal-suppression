#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.data.loaders import load_prompt_examples
from frs.training.qlora import QLoRAConfig
from frs.training.unsloth_runner import build_unsloth_dataset, run_unsloth_training
from frs.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Write a scaffold training artifact for QLoRA repair runs.')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    examples = load_prompt_examples(args.prompts)
    config = QLoRAConfig(model_id=args.model_id)
    dataset = build_unsloth_dataset(examples)
    result = run_unsloth_training(config, dataset)
    write_json(args.output, result)
    print(args.output)


if __name__ == '__main__':
    main()
