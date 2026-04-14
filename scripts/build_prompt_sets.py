#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.data.loaders import load_prompt_examples, write_prompt_examples
from frs.data.prompts import group_counts


def main() -> None:
    parser = argparse.ArgumentParser(description='Normalize an input JSONL prompt file into project schema.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    examples = load_prompt_examples(args.input)
    write_prompt_examples(args.output, examples)
    print(group_counts(examples))


if __name__ == '__main__':
    main()
