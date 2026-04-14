#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.editing.apply_edit import EditSpec, apply_directional_ablation
from frs.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply a directional ablation to a matrix artifact.')
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--direction', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--norm-preserving', action='store_true')
    args = parser.parse_args()

    matrix = read_json(args.matrix)
    direction = read_json(args.direction)
    edited = apply_directional_ablation(matrix, direction, EditSpec(strength=args.strength, norm_preserving=args.norm_preserving))
    write_json(args.output, edited)
    print(args.output)


if __name__ == '__main__':
    main()
