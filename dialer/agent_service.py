"""Agent presence: login, logout, pause, wrap-up timers and sudden disappearance."""
from collections import Counter
from typing import Optional

from core.clock import Clock
from core.database import Database
from core.lock_manager import LockManager
from core.models import Agent
from core.state_machines import AgentState, CallState, PRE_CONNECT_CALL_STATES
from dialer.call_lifecycle import CallLifecycle
from dialer.safety_controller import SafeDialGateway


class AgentService:
    def __init__(self, db: Database, locks: LockManager, lifecycle: CallLifecycle,
                 gateway: SafeDialGateway, clock: Clock):
        self.db = db
        self.locks = locks
        self.lifecycle = lifecycle
        self.gateway = gateway
        self.clock = clock
        self.stats: Counter = Counter()

    def add_agent(self, agent_id: str, name: Optional[str] = None, logged_in: bool = True) -> None:
        self.db.insert_agent(Agent(id=agent_id, name=name or agent_id, state=AgentState.OFFLINE,
                                   state_changed_at=self.clock.now()))
        if logged_in:
            self.login(agent_id)

    def login(self, agent_id: str) -> bool:
        return self.locks.transition_agent(agent_id, AgentState.AVAILABLE, from_states={AgentState.OFFLINE},
                                           changes={"logout_requested": False})

    def pause(self, agent_id: str) -> bool:
        return self.locks.transition_agent(agent_id, AgentState.PAUSED,
                                           from_states={AgentState.AVAILABLE, AgentState.WRAP_UP})

    def logout(self, agent_id: str) -> bool:
        """Graceful: immediate if idle, otherwise after the current call's wrap-up."""
        if self.locks.transition_agent(agent_id, AgentState.OFFLINE,
                                       from_states={AgentState.AVAILABLE, AgentState.PAUSED}):
            return True
        agent = self.db.get_agent(agent_id)
        return agent is not None and agent.state != AgentState.OFFLINE and self.db.cas_update(
            "agents", agent.id, agent.version, {"logout_requested": True})

    def complete_wrapups(self) -> int:
        now = self.clock.now()
        done = 0
        for agent in self.db.list_agents([AgentState.WRAP_UP]):
            if agent.wrap_until is not None and agent.wrap_until > now:
                continue
            target = AgentState.OFFLINE if agent.logout_requested else AgentState.AVAILABLE
            if self.locks.transition_agent(agent.id, target, from_states={AgentState.WRAP_UP},
                                           changes={"wrap_until": None, "logout_requested": False}):
                done += 1
        return done

    def force_offline(self, agent_id: str) -> None:
        """Agent vanished (browser crash, network drop). Takes effect immediately.

        * pre-answer call bound to them: unbind it. If it is answered, the bridge
          finds another idle agent or abandons it after the hold time.
        * call in progress: end it (the borrower gets a callback later).
        """
        hangup_pcid = None
        for _ in range(LockManager.MAX_CAS_RETRIES):
            agent = self.db.get_agent(agent_id)
            if agent is None or agent.state == AgentState.OFFLINE:
                return
            call_id = agent.reserved_for_call_id
            with self.db.transaction():
                if not self.db.cas_update("agents", agent.id, agent.version,
                                          {"state": AgentState.OFFLINE, "reserved_for_call_id": None,
                                           "wrap_until": None, "logout_requested": False,
                                           "state_changed_at": self.clock.now()}):
                    continue
                call = self.db.get_call(call_id) if call_id else None
                if call is not None and call.agent_id == agent.id:
                    if call.state in PRE_CONNECT_CALL_STATES:
                        self.locks.update_call_fields(call.id, {"agent_id": None},
                                                      only_in_states=PRE_CONNECT_CALL_STATES)
                        self.stats["calls_unbound"] += 1
                    elif call.state == CallState.CONNECTED:
                        if self.lifecycle.finalize(call.id, CallState.COMPLETED, "agent_lost"):
                            hangup_pcid = call.provider_call_id
                            self.stats["calls_dropped"] += 1
            self.stats["forced_offline"] += 1
            break
        if hangup_pcid:
            self.gateway.hangup(hangup_pcid)
