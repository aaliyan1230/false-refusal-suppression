from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from frs.data.schemas import PromptExample


def load_prompt_examples(path: str) -> List[PromptExample]:
    records = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(PromptExample.from_dict(json.loads(line)))
    return records


def write_prompt_examples(path: str, examples: Iterable[PromptExample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True) + '\n')
