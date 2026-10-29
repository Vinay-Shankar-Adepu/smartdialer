"""Progressive pacing: one call per idle agent, and the agent is reserved BEFORE dialing.

Abandonment is impossible by construction, because every call that rings
already owns an agent. The cost is that the agent sits idle for the whole
ring time and for every unanswered attempt.
"""
from dialer.pacing import PROGRESSIVE, PacingDecision
from dialer.snapshot import Snapshot


class ProgressivePacing:
    mode = PROGRESSIVE

    def recommend(self, snap: Snapshot) -> PacingDecision:
        idle = snap.idle_agents
        return PacingDecision(
            mode=PROGRESSIVE,
            recommended=idle,
            summary=f"Progressive: {idle} idle agent(s) -> dial {idle} (each call reserves its agent first)",
            explanation={"idle_agents": idle, "formula": "calls = idle agents"},
        )
