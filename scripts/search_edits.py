#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.editing.search import EditCandidate, rank_candidates
from frs.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank edit candidates using calibration score.')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    raw_candidates = read_json(args.candidates)
    candidates = [EditCandidate(**payload) for payload in raw_candidates]
    ranked = [candidate.__dict__ for candidate in rank_candidates(candidates)]
    write_json(args.output, ranked)
    print(args.output)


if __name__ == '__main__':
    main()
