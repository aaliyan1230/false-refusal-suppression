#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Write a measurement artifact stub for activation capture runs.')
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--split-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', default='4bit_fp32_stats')
    args = parser.parse_args()

    artifact = {
        'model_id': args.model_id,
        'split_path': args.split_path,
        'precision': args.precision,
        'status': 'scaffold_stub',
    }
    write_json(args.output, artifact)
    print(args.output)


if __name__ == '__main__':
    main()
