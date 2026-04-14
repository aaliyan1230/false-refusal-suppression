from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class CalibrationMetrics:
    false_refusal_rate: float
    true_refusal_rate: float
    capability_retention: float
    harmless_kl_penalty: float

    def calibration_score(self, w1: float = 1.0, w2: float = 1.0, w3: float = 1.0, w4: float = 1.0) -> float:
        return (
            w1 * (1.0 - self.false_refusal_rate)
            + w2 * self.true_refusal_rate
            + w3 * self.capability_retention
            - w4 * self.harmless_kl_penalty
        )

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload['calibration_score'] = self.calibration_score()
        return payload
