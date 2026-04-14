from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    return logging.getLogger(name)
