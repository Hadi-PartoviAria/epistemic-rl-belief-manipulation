"""Reward Machine — Finite automaton tracking magic trick phases.

Implements the Reward Machine from the report (with limes enabled):
    u0 (Setup) → u1 (Covered) → u2 (Pocket) → u3 (Appear)
    u3 → u0 (loop back) | u3 → uT (ball milestone)
    u3 → u4 (lime loaded) → uT (lime shown + milestone)
    uT → u4 (loop back to load more limes)

Note: there is NO terminal RM state — episodes end only when the step limit is
reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

from epistemic_logic import EpistemicState


class RMState(Enum):
    """Reward Machine states."""
    U0 = "u0"      # Setup: initial state
    U1 = "u1"      # Covered: observer saw ball under cup (Δ ball_cup)
    U2 = "u2"      # Pocket: observer infers ball in hand (□ ball_hand)
    U3 = "u3"      # Appear: ball is visible under cup (ball_visible)
    U4 = "u4"      # Lime Loaded: lime loaded, observer believes no lime
    UT = "uT"      # Lime Shown / Milestone (not terminal — can loop to u4)


@dataclass
class RMTransition:
    """A transition in the Reward Machine."""
    source: RMState
    target: RMState
    label: str                                    # descriptive label (a, b, c, ...)
    condition: Callable[[EpistemicState], bool]    # propositional/epistemic guard
    reward: float                                 # reward granted on this transition
    description: str = ""


class RewardMachine:
    """Reward Machine automaton for the Chop Cup magic trick.

    The RM encodes the temporal structure of the magic routine,
    providing shaped rewards for each successful phase transition.

    Key insight: the RM state is part of the RL state, making the
    temporal structure of deception Markovian.
    """

    def __init__(self, enable_limes: bool = True, shaping: bool = False):
        self.state: RMState = RMState.U0
        self.enable_limes = enable_limes
        self.shaping = shaping  # if True, give small intermediate rewards for progress
        self.transitions: List[RMTransition] = self._build_transitions()
        self.total_reward: float = 0.0
        self.transition_history: List[str] = []
        self.reached_states: set = set()  # track which RM states were visited

    def _build_transitions(self) -> List[RMTransition]:
        """Build all RM transitions with epistemic conditions.

        Transition labels follow the report notation:
          a: Δ(ball_cup)         — observer sees ball under cup
          b: □(ball_hand)        — observer infers ball in hand
          h: ball_visible        — ball appears (reveal)
          c: cup_down            — reset (cup placed back down)
          g: ball_only           — ball milestone (can continue to limes)
          k: lime_visible        — lime shown + milestone
          f: □(¬lime_cup)        — loop back to load more limes
        """
        transitions = []

        # Shaping reward: small bonus for making progress through RM phases
        shape = 0.1 if self.shaping else 0.0

        # ---- Ball Phase ----

        # a: u0 → u1 — Observer sees ball under cup
        transitions.append(RMTransition(
            source=RMState.U0, target=RMState.U1, label="a",
            condition=lambda s: s.has("eb_reg_under"),
            reward=shape,
            description="Observer sees ball under cup (Δ ball_cup)",
        ))

        # b: u1 → u2 — Observer infers ball is in hand (SAT deduction!)
        transitions.append(RMTransition(
            source=RMState.U1, target=RMState.U2, label="b",
            condition=lambda s: s.implicit_belief("reg_hand", positive=True),
            reward=shape,
            description="Observer infers ball in hand (□ ball_hand)",
        ))

        # h: u2 → u3 — Ball appears (magnetic ball revealed)
        transitions.append(RMTransition(
            source=RMState.U2, target=RMState.U3, label="h",
            condition=lambda s: s.has("ball_visible"),
            reward=shape,
            description="Ball magically appears! (Appear effect)",
        ))

        # ---- From U3: milestone or reset ----
        # (Lime phase is entered via uT → u4 using transition f)

        # c: u3 → u0 — Reset for another cycle (no reward)
        transitions.append(RMTransition(
            source=RMState.U3, target=RMState.U0, label="c",
            condition=lambda s: s.has("cup_down"),
            reward=0.0,
            description="Cup placed down, reset for another cycle",
        ))

        # g: u3 → uT — Ball deception milestone — MAIN REWARD
        transitions.append(RMTransition(
            source=RMState.U3, target=RMState.UT, label="g",
            condition=lambda s: s.has("ball_visible") and not s.has("lime_visible"),
            reward=1.0,
            description="DECEPTION SUCCESS — Observer shocked! +1",
        ))

        # ---- Lime Finale Phase ----
        if self.enable_limes:
            # k: u4 → uT — Show lime AND count it as the milestone
            transitions.append(RMTransition(
                source=RMState.U4, target=RMState.UT, label="k",
                condition=lambda s: s.has("lime_visible"),
                reward=10.0,
                description="DECEPTION SUCCESS — Lime reveal! +10",
            ))

            # f: uT → u4 — Loop back to load more limes
            # After a successful reveal (ball or lime), the agent can continue.
            transitions.append(RMTransition(
                source=RMState.UT, target=RMState.U4, label="f",
                condition=lambda s: s.implicit_belief("lime_cup", positive=False),
                reward=0.0,
                description="Loop back — load another lime",
            ))

        return transitions

    def reset(self) -> None:
        """Reset the RM to initial state."""
        self.state = RMState.U0
        self.total_reward = 0.0
        self.transition_history = []
        self.reached_states = {RMState.U0}

    def step(self, env_state: EpistemicState) -> Tuple[float, bool, str]:
        """Evaluate RM transitions given the current environment state.

        Returns:
            (reward, is_terminal, transition_description)
        """
        for t in self.transitions:
            if t.source == self.state and t.condition(env_state):
                old_state = self.state
                self.state = t.target
                self.total_reward += t.reward
                self.reached_states.add(t.target)
                desc = f"{old_state.value} --[{t.label}]--> {t.target.value}: {t.description}"
                self.transition_history.append(desc)
                return t.reward, self.is_terminal(), desc

        return 0.0, self.is_terminal(), ""

    def query_transition(self, from_state: RMState, env_state: EpistemicState) -> Tuple[RMState, float]:
        """Non-mutating query: what transition would fire from `from_state` given `env_state`?

        Used by QRM for counterfactual experience replay — simulates RM
        transitions from ANY RM state without changing the actual RM state.

        Returns (next_rm_state, reward). If no transition fires, returns (from_state, 0.0).
        """
        for t in self.transitions:
            if t.source == from_state and t.condition(env_state):
                return t.target, t.reward
        return from_state, 0.0

    def get_all_states(self) -> List[RMState]:
        """Return all non-terminal RM states (for QRM iteration)."""
        if self.enable_limes:
            return [RMState.U0, RMState.U1, RMState.U2, RMState.U3, RMState.U4, RMState.UT]
        else:
            return [RMState.U0, RMState.U1, RMState.U2, RMState.U3, RMState.UT]

    def is_terminal(self) -> bool:
        """Check if the RM is in a terminal state.

        There is no terminal state — episodes end only by step limit.
        """
        return False

    def get_state(self) -> RMState:
        return self.state

    def get_state_index(self) -> int:
        """Return numeric index for the RM state (for Q-table)."""
        return list(RMState).index(self.state)

    def __repr__(self) -> str:
        return f"RM(state={self.state.value}, reward={self.total_reward:.1f})"
