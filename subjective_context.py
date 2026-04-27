"""
Subjective (internal) context for feature engineering: sleep, fatigue, dietary/social state.
Complements external context (time, weather) in the multi-dimensional state space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class SubjectiveContext:
    """User-reported internal state. Bounds match UI and ML feature expectations."""

    sleep_hours: float = 7.0
    fatigue: int = 5
    lactose_intolerance: bool = False
    social_battery: str = "Full"

    def __post_init__(self) -> None:
        self.sleep_hours = float(max(0.0, min(12.0, self.sleep_hours)))
        self.fatigue = int(max(1, min(10, self.fatigue)))
        sb = (self.social_battery or "Full").strip()
        self.social_battery = "Empty" if sb.lower() == "empty" else "Full"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "SubjectiveContext":
        if not d:
            return SubjectiveContext()
        return SubjectiveContext(
            sleep_hours=float(d.get("sleep_hours", 7.0)),
            fatigue=int(d.get("fatigue", 5)),
            lactose_intolerance=bool(d.get("lactose_intolerance", False)),
            social_battery=str(d.get("social_battery", "Full")),
        )

    def feature_vector(self) -> Tuple[float, float, float, float]:
        """
        Numerical + encoded categorical features for a classifier.
        (sleep, fatigue, lactose as 0/1, social as 0/1 Empty->1)
        """
        return (
            self.sleep_hours,
            float(self.fatigue),
            1.0 if self.lactose_intolerance else 0.0,
            1.0 if self.social_battery == "Empty" else 0.0,
        )
