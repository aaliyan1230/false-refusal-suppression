from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def render_text_report(metrics: Mapping[str, float]) -> str:
    lines = ['False-Refusal Suppression Evaluation Summary']
    for key, value in metrics.items():
        lines.append(f'- {key}: {value}')
    return '\n'.join(lines)


def write_json_report(path: str, metrics: Mapping[str, float]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True) + '\n', encoding='utf-8')
