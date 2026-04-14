#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from frs.data.manifests import create_split_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description='Create grouped discovery/selection/holdout splits.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    summary = create_split_manifests(
        input_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        seed=args.seed,
    )
    print(summary)


if __name__ == '__main__':
    main()
