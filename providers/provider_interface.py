"""Abstract telecom provider.

Deliberate differences from a naive `initiate_call(agent, borrower)`:

* The provider never sees the agent. In predictive mode a call is placed
  before any agent is chosen, so agent selection must stay on our side.
* We pass OUR call id as the provider's client reference (idempotency key).
  A retried initiate_call with the same id returns the same provider call
  instead of dialing the borrower twice. `lookup(call_id)` lets crash recovery
  ask "did that dial actually go out?"
* Events are pulled with poll_events(). In production this is a webhook
  receiver writing to a queue; the processing semantics are the same.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from core.models import ProviderEvent


class ProviderError(Exception):
    """Provider rejected or failed the request."""


class ProviderTimeout(ProviderError):
    """No response in time. The call MAY OR MAY NOT have been placed."""


@dataclass(frozen=True)
class ProviderCallHandle:
    provider_call_id: str
    call_id: str
    provider: str


@dataclass(frozen=True)
class ProviderCallStatus:
    provider_call_id: str
    call_id: str
    is_live: bool


class TelecomProvider(ABC):
    name: str = "ABSTRACT"

    @abstractmethod
    def initiate_call(self, call_id: str, phone: str) -> ProviderCallHandle:
        """Start dialing `phone`. Idempotent on `call_id`."""

    @abstractmethod
    def hangup(self, provider_call_id: str) -> None:
        """Terminate a call. Safe to call on an already-ended call."""

    @abstractmethod
    def lookup(self, call_id: str) -> Optional[ProviderCallStatus]:
        """Find a call by our client reference. None = provider never placed it."""

    @abstractmethod
    def poll_events(self) -> List[ProviderEvent]:
        """Return events delivered since the last poll (possibly duplicated or out of order)."""
