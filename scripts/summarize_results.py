#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Print a compact summary of a JSON artifact.')
    parser.add_argument('--input', required=True)
    args = parser.parse_args()

    payload = read_json(args.input)
    print(payload)


if __name__ == '__main__':
    main()
