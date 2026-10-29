"""Common types for pacing engines.

A pacing engine is a pure function: Snapshot -> PacingDecision. It has no
database handle, no provider and no allocator, so it cannot dial anything. It
can only make a recommendation that the SafetyController judges.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

from dialer.snapshot import Snapshot

PROGRESSIVE = "progressive"
PREDICTIVE = "predictive"


@dataclass
class PacingDecision:
    mode: str
    recommended: int
    summary: str                                   # one human-readable sentence
    explanation: Dict[str, Any] = field(default_factory=dict)  # every input and intermediate value


class PacingEngine(Protocol):
    mode: str

    def recommend(self, snap: Snapshot) -> PacingDecision:
        ...
