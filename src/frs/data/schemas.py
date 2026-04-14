from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

ALLOWED_GROUPS = {
    "benign_easy",
    "benign_borderline",
    "unsafe_true_refusal",
    "capability_holdout",
}

ALLOWED_EXPECTED_BEHAVIORS = {"answer", "refuse"}


@dataclass(frozen=True)
class PromptExample:
    id: str
    prompt: str
    group: str
    topic: str
    expected_behavior: str
    source: str
    notes: Optional[str] = None
    family_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("PromptExample.id must be non-empty")
        if not self.prompt.strip():
            raise ValueError("PromptExample.prompt must be non-empty")
        if self.group not in ALLOWED_GROUPS:
            raise ValueError(f"Unsupported group: {self.group}")
        if self.expected_behavior not in ALLOWED_EXPECTED_BEHAVIORS:
            raise ValueError(f"Unsupported expected_behavior: {self.expected_behavior}")

    @property
    def resolved_family_id(self) -> str:
        return self.family_id or self.id

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PromptExample":
        metadata = dict(payload.get("metadata") or {})
        example = cls(
            id=payload["id"],
            prompt=payload["prompt"],
            group=payload["group"],
            topic=payload["topic"],
            expected_behavior=payload["expected_behavior"],
            source=payload["source"],
            notes=payload.get("notes"),
            family_id=payload.get("family_id"),
            metadata=metadata,
        )
        example.validate()
        return example

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "group": self.group,
            "topic": self.topic,
            "expected_behavior": self.expected_behavior,
            "source": self.source,
        }
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.family_id is not None:
            payload["family_id"] = self.family_id
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload
