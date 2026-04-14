from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from frs.data.config import load_prompt_set_config
from frs.data.loaders import load_prompt_examples, write_prompt_examples
from frs.data.splits import make_grouped_splits, summarize_splits


def create_split_manifests(
    input_path: str,
    output_dir: str,
    config_path: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, int]]:
    examples = load_prompt_examples(input_path)
    split_group_targets = None
    if config_path:
        split_group_targets = load_prompt_set_config(config_path).split_group_targets

    splits = make_grouped_splits(examples, split_group_targets=split_group_targets, seed=seed)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits.items():
        write_prompt_examples(str(resolved_output_dir / f'{split_name}.jsonl'), split_examples)

    return summarize_splits(splits)
