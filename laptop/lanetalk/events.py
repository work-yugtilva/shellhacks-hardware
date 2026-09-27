"""Structured events for later LaneTalk pipeline stages."""

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass(slots=True)
class Event:
    kind: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
