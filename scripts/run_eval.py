#!/usr/bin/env python3
from __future__ import annotations

import argparse

from frs.evaluation.metrics import CalibrationMetrics
from frs.evaluation.reports import render_text_report, write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(description='Write an evaluation report artifact from scalar metrics.')
    parser.add_argument('--false-refusal-rate', type=float, required=True)
    parser.add_argument('--true-refusal-rate', type=float, required=True)
    parser.add_argument('--capability-retention', type=float, required=True)
    parser.add_argument('--harmless-kl-penalty', type=float, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    metrics = CalibrationMetrics(
        false_refusal_rate=args.false_refusal_rate,
        true_refusal_rate=args.true_refusal_rate,
        capability_retention=args.capability_retention,
        harmless_kl_penalty=args.harmless_kl_penalty,
    )
    payload = metrics.to_dict()
    write_json_report(args.output, payload)
    print(render_text_report(payload))


if __name__ == '__main__':
    main()
