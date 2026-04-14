#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.editing.directions import direction_from_contrast, separability_score
from frs.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute a normalized direction from two vector groups.')
    parser.add_argument('--group-a', required=True)
    parser.add_argument('--group-b', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    group_a = read_json(args.group_a)
    group_b = read_json(args.group_b)
    payload = {
        'direction': direction_from_contrast(group_a, group_b),
        'separability_score': separability_score(group_a, group_b),
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == '__main__':
    main()
