"""Small, explainable statistical helpers shared by pacing and safety."""
import math


def smoothed_rate(successes: float, trials: float, prior_rate: float, prior_weight: float) -> float:
    """Beta-prior smoothing: behaves like `prior_weight` imaginary calls at
    `prior_rate`, so 3 answers out of 3 calls does not read as 100%."""
    return (successes + prior_rate * prior_weight) / (trials + prior_weight)


def wilson_upper(successes: float, trials: float, z: float) -> float:
    """Upper bound of the Wilson score interval for a proportion.

    The dialer uses the UPPER bound of the answer rate on purpose.
    Overestimating the answer rate makes us dial FEWER lines, which is the
    safe direction to be wrong in.
    """
    if trials <= 0:
        return 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return min(1.0, (centre + margin) / denom)


def lines_for_free_agents(free_agents: float, p: float, k_sigma: float) -> float:
    """Largest number of lines N such that answers stay within free agents
    with a k-sigma margin:

        N*p + k * sqrt(N*p*(1-p)) <= free_agents

    Substituting x = sqrt(N) gives  p*x^2 + b*x - free = 0  with
    b = k*sqrt(p*(1-p)), so  x = (-b + sqrt(b^2 + 4*p*free)) / (2p).
    With k=0 this reduces to the naive free/p.
    """
    if free_agents <= 0 or p <= 0:
        return 0.0
    b = k_sigma * math.sqrt(p * (1 - p))
    x = (-b + math.sqrt(b * b + 4 * p * free_agents)) / (2 * p)
    return x * x


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
