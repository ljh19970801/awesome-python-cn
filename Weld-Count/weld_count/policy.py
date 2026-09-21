from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionPolicy:
    """A conservative policy: OK only when count is above the configured limit."""
    min_score: float = 0.5
    # 16 -> 8 means count > 8 (at least 9); 16 -> 9 means count > 9.
    strict_majority: bool = True
    limits: dict[int, int] | None = None

    def limit(self, total: int) -> int:
        if self.limits and total in self.limits:
            return int(self.limits[total])
        if total <= 0:
            raise ValueError("total must be positive")
        return total // 2

    def decide_count(self, count: int, total: int) -> str:
        # strict_majority is explicit to avoid the common > versus >= ambiguity.
        threshold = self.limit(total)
        return "OK" if count > threshold else "NG"

    def decide(self, detections: list[dict[str, Any]], total: int) -> dict[str, Any]:
        accepted = [d for d in detections if float(d.get("score", 1.0)) >= self.min_score]
        return {"count": len(accepted), "total": total,
                "decision": self.decide_count(len(accepted), total),
                "threshold": self.limit(total), "min_score": self.min_score}

    def to_dict(self) -> dict[str, Any]:
        return {"min_score": self.min_score, "strict_majority": self.strict_majority,
                "limits": self.limits or {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionPolicy":
        return cls(float(data.get("min_score", .5)), bool(data.get("strict_majority", True)),
                   {int(k): int(v) for k, v in data.get("limits", {}).items()})
