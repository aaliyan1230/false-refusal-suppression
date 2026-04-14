from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional fallback for minimal environments
    yaml = None


@dataclass(frozen=True)
class PromptSetConfig:
    groups: Dict[str, Dict[str, Any]]
    split_group_targets: Dict[str, Dict[str, int]]
    rules: Dict[str, Any]

    @property
    def expected_behavior_by_group(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for group, payload in self.groups.items():
            if 'expected_behavior' in payload:
                mapping[group] = str(payload['expected_behavior'])
        return mapping


def load_prompt_set_config(path: str) -> PromptSetConfig:
    payload = _load_yaml_payload(Path(path).read_text(encoding='utf-8'))
    groups = _ensure_mapping(payload.get('groups'), 'groups')
    splits = _ensure_mapping(payload.get('splits'), 'splits')
    rules = _ensure_mapping(payload.get('rules', {}), 'rules')

    split_group_targets: Dict[str, Dict[str, int]] = {}
    for split_name, group_counts in splits.items():
        counts_mapping = _ensure_mapping(group_counts, f'splits.{split_name}')
        split_group_targets[split_name] = {
            str(group): int(count)
            for group, count in counts_mapping.items()
        }

    normalized_groups = {
        str(group): _ensure_mapping(group_payload, f'groups.{group}')
        for group, group_payload in groups.items()
    }
    normalized_rules = {str(key): value for key, value in rules.items()}

    return PromptSetConfig(
        groups=normalized_groups,
        split_group_targets=split_group_targets,
        rules=normalized_rules,
    )


def _ensure_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'{field_name} must be a mapping')
    return dict(value)


def _load_yaml_payload(text: str) -> Dict[str, Any]:
    if yaml is not None:
        payload = yaml.safe_load(text) or {}
        if not isinstance(payload, dict):
            raise ValueError('Top-level prompt set config must be a mapping')
        return dict(payload)
    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, MutableMapping[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith('#'):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(' '))
        stripped = raw_line.strip()
        if ':' not in stripped:
            raise ValueError(f'Unsupported YAML line: {raw_line}')
        key, raw_value = stripped.split(':', 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if raw_value == '':
            next_mapping: Dict[str, Any] = {}
            current[key] = next_mapping
            stack.append((indent, next_mapping))
            continue

        current[key] = _parse_scalar(raw_value)

    return root


def _parse_scalar(value: str) -> Any:
    if value == '{}':
        return {}
    if value == 'true':
        return True
    if value == 'false':
        return False
    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        return int(value)
    return value
