#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from frs.data.loaders import load_prompt_examples, write_prompt_examples
from frs.data.splits import make_grouped_splits, summarize_splits


def main() -> None:
    parser = argparse.ArgumentParser(description='Create grouped discovery/selection/holdout splits.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    examples = load_prompt_examples(args.input)
    splits = make_grouped_splits(examples, seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits.items():
        write_prompt_examples(str(output_dir / f'{split_name}.jsonl'), split_examples)

    print(summarize_splits(splits))


if __name__ == '__main__':
    main()
