"""Provider A: fast and reliable.

* 0.1 s request latency
* 95% of dials are carried (5% fail with a FAILED event)
* events delivered exactly once, in order
"""
from providers.mock_base import MockProvider


class ProviderA(MockProvider):
    name = "PROVIDER_A"
    latency_s = 0.1
    failure_rate = 0.05
