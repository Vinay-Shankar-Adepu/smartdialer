"""Clock abstraction.

Every timestamp in the system comes from a Clock, never from time.time()
directly. That lets the same code run against wall-clock time in production
and against a deterministic simulated clock in tests and simulations (an hour
of dialing simulates in seconds, and failure scenarios are reproducible).
"""
import threading
import time
from abc import ABC, abstractmethod


class Clock(ABC):
    @abstractmethod
    def now(self) -> float:
        """Current time in seconds (epoch seconds or simulated seconds)."""


class SystemClock(Clock):
    def now(self) -> float:
        return time.time()


class SimClock(Clock):
    """Manually advanced clock. Thread-safe so concurrent tests can share it."""

    def __init__(self, start: float = 0.0):
        self._now = float(start)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("time only moves forward")
        with self._lock:
            self._now += seconds
            return self._now
