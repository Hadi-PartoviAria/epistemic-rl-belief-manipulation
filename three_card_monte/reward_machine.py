"""Reward Machine for the Three-Card Monte trick.

RM states:
    u0 (Setup)    → u1 (Revealed)  → u2 (Shuffled) → u3 (Deceived, +1.0)
    u2 self-loop (d): additional shuffles
    u3 → u0 (e): reset for another round

No terminal RM state — episodes end only when the step limit is reached.
The agent can loop u0→u1→u2→u3→u0→... to accumulate +1.0 per deception.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from epistemic_logic import EpistemicState


class RMState(Enum):
    """Reward Machine states for Three-Card Monte."""
    U0 = "u0"   # Setup: cards face-down, no info given
    U1 = "u1"   # Revealed: observer saw money card position
    U2 = "u2"   # Shuffled: at least one swap occurred
    U3 = "u3"   # Deceived: false belief achieved (+1.0 milestone)


@dataclass
class RMTransition:
    """A transition in the Reward Machine."""
    source: RMState
    target: RMState
    label: str
    condition: Callable[[EpistemicState], bool]
    reward: float
    description: str = ""


def _observer_has_false_belief(s: EpistemicState) -> bool:
    """Check if observer holds a false belief about the money card position.

    Returns True iff ∃ p ∈ {L, C, R}: □(money_p) ∧ ¬money_p
    i.e. the SAT solver deduces money is at p, but it actually isn't.
    """
    for pos in ("money_L", "money_C", "money_R"):
        if s.implicit_belief(pos, positive=True) and not s.has(pos):
            return True
    return False


class RewardMachine:
    """Reward Machine for Three-Card Monte.

    Encodes the temporal structure of a deception round:
      u0 → u1 (show money card) → u2 (shuffle) → u3 (false belief achieved, +1)
      u3 → u0 (reset for another round)
    """

    def __init__(self):
        self.state: RMState = RMState.U0
        self.transitions: List[RMTransition] = self._build_transitions()
        self.total_reward: float = 0.0
        self.transition_history: List[str] = []
        self.reached_states: set = set()

    def _build_transitions(self) -> List[RMTransition]:
        """Build all RM transitions with epistemic conditions."""
        transitions = []

        # a: u0 → u1 — Observer sees the money card
        transitions.append(RMTransition(
            source=RMState.U0, target=RMState.U1, label="a",
            condition=lambda s: s.has("face_up") and any(
                s.has(f"eb_{pos}") and s.has(pos)
                for pos in ("money_L", "money_C", "money_R")
            ),
            reward=0.0,
            description="Observer sees the money card (face up + correct belief)",
        ))

        # b: u1 → u2 — Cards hidden and shuffle begins
        transitions.append(RMTransition(
            source=RMState.U1, target=RMState.U2, label="b",
            condition=lambda s: not s.has("face_up") and s.has("cards_moving"),
            reward=0.0,
            description="Cards face-down and shuffle begins",
        ))

        # c: u2 → u3 — False belief achieved (epistemic core!)
        transitions.append(RMTransition(
            source=RMState.U2, target=RMState.U3, label="c",
            condition=_observer_has_false_belief,
            reward=1.0,
            description="Deception success! Observer holds false belief (+1.0)",
        ))

        # d: u2 → u2 — Additional shuffles (self-loop)
        transitions.append(RMTransition(
            source=RMState.U2, target=RMState.U2, label="d",
            condition=lambda s: s.has("cards_moving"),
            reward=0.0,
            description="Additional swap during shuffle phase",
        ))

        # e: u3 → u0 — Reset for another round
        transitions.append(RMTransition(
            source=RMState.U3, target=RMState.U0, label="e",
            condition=lambda s: s.has("face_up"),
            reward=0.0,
            description="Cards shown, reset for new round",
        ))

        return transitions

    def reset(self) -> None:
        """Reset RM to initial state."""
        self.state = RMState.U0
        self.total_reward = 0.0
        self.transition_history = []
        self.reached_states = {RMState.U0}

    def step(self, env_state: EpistemicState) -> Tuple[float, bool, str]:
        """Evaluate RM transitions given the current environment state.

        Returns (reward, terminal, info_string).
        Terminal is always False — episodes end by step limit.
        """
        for t in self.transitions:
            if t.source == self.state and t.condition(env_state):
                old = self.state
                self.state = t.target
                self.total_reward += t.reward
                self.transition_history.append(t.label)
                self.reached_states.add(t.target)
                info = f"RM: {old.value}→{t.target.value} ({t.label}) r={t.reward}"
                return t.reward, False, info

        return 0.0, False, ""

    def query_transition(
        self, from_state: RMState, env_state: EpistemicState
    ) -> Tuple[RMState, float]:
        """Non-mutating query: what would happen from a given RM state?

        Used by QRM for counterfactual replay.
        """
        for t in self.transitions:
            if t.source == from_state and t.condition(env_state):
                return t.target, t.reward
        return from_state, 0.0

    def get_all_states(self) -> List[RMState]:
        """Return all RM states."""
        return [RMState.U0, RMState.U1, RMState.U2, RMState.U3]

    def get_state(self) -> RMState:
        return self.state

    def get_state_index(self) -> int:
        """Numeric index for Q-table / neural net input."""
        idx_map = {RMState.U0: 0, RMState.U1: 1, RMState.U2: 2, RMState.U3: 3}
        return idx_map[self.state]

    def is_terminal(self) -> bool:
        """RM never terminates — episodes end by step limit."""
        return False
