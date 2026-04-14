from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def read_env_value(name: str, dotenv_path: str | None = None) -> str | None:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    if dotenv_path is None:
        return None
    return load_dotenv_file(dotenv_path).get(name)