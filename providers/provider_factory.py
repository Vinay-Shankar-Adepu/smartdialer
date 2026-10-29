"""Construct providers by short name, so switching is a config change."""
from typing import Dict, Optional, Type

from core.clock import Clock
from providers.mock_base import BorrowerBehavior, MockProvider
from providers.provider_a import ProviderA
from providers.provider_b import ProviderB

PROVIDERS: Dict[str, Type[MockProvider]] = {"A": ProviderA, "B": ProviderB}


def create_provider(name: str, clock: Clock, behavior: BorrowerBehavior,
                    seed: Optional[int] = None) -> MockProvider:
    key = name.upper().replace("PROVIDER_", "")
    try:
        cls = PROVIDERS[key]
    except KeyError:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}") from None
    return cls(clock, behavior, seed=seed)
