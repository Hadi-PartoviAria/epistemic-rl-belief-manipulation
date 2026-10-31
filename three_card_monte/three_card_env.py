"""Three-Card Monte Environment — Epistemic RL for card deception.

The dealer (agent) must manipulate the observer's beliefs about the money card
location through a combination of fair swaps, fake swaps, and secret swaps.

Physical fluents (5):
    money_L, money_C, money_R  — actual money card position (exactly one true)
    face_up                    — cards are currently visible
    cards_moving               — a swap just occurred

Epistemic fluents (6):
    eb_money_L, eb_not_money_L — observer believes/disbelieves money at Left
    eb_money_C, eb_not_money_C — observer believes/disbelieves money at Centre
    eb_money_R, eb_not_money_R — observer believes/disbelieves money at Right

Actions (5):
    show_cards     — flip cards face-up (honest reveal)
    hide_cards     — flip cards face-down
    fair_swap_XY   — swap X↔Y, observer tracks it
    fake_swap_XY   — observer thinks X↔Y happened, but cards don't move
    secret_swap_XY — cards swap X↔Y, but observer doesn't notice
    (X,Y pairs: LC, LR, CR)
"""

from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from epistemic_logic import CommonGround, EpistemicState
from reward_machine import RewardMachine, RMState

# All possible actions
ALL_ACTIONS = [
    "show_cards",
    "hide_cards",
    "fair_swap_LC", "fair_swap_LR", "fair_swap_CR",
    "fake_swap_LC", "fake_swap_LR", "fake_swap_CR",
    "secret_swap_LC", "secret_swap_LR", "secret_swap_CR",
]

# Position pairs for swap actions
SWAP_PAIRS = {
    "LC": ("money_L", "money_C"),
    "LR": ("money_L", "money_R"),
    "CR": ("money_C", "money_R"),
}


class ThreeCardMonteEnv:
    """Three-Card Monte environment with epistemic reasoning."""

    def __init__(self, max_steps: int = 50):
        self.max_steps = max_steps
        self.step_count = 0

        # Common Ground: money card is at exactly one position
        self.common_ground = CommonGround(
            variables=["money_L", "money_C", "money_R"],
            exactly_one_groups=[["money_L", "money_C", "money_R"]],
        )

        self.state: Optional[EpistemicState] = None
        self.reward_machine = RewardMachine()
        self.done = False

    def reset(self, money_position: Optional[str] = None) -> Tuple[EpistemicState, RMState]:
        """Reset the environment.

        Args:
            money_position: Force money card position ('money_L', 'money_C', 'money_R').
                          If None, randomly chosen.
        """
        self.step_count = 0
        self.done = False
        self.reward_machine.reset()

        # Choose money card position
        if money_position is None:
            money_position = random.choice(["money_L", "money_C", "money_R"])

        # Initial state: cards face-down, no observer beliefs yet
        initial_fluents: Set[str] = {money_position}
        # Cards start face-down (face_up is absent = False)
        # No belief fluents set yet — observer has no info

        self.state = EpistemicState(
            fluents=frozenset(initial_fluents),
            common_ground=self.common_ground,
        )

        return self.state, self.reward_machine.get_state()

    def get_available_actions(self) -> List[str]:
        """Return currently available actions based on state guards."""
        s = self.state
        available = []

        face_up = s.has("face_up")

        if not face_up:
            # Cards are face-down: can show, or do swaps
            available.append("show_cards")
            for pair in SWAP_PAIRS:
                available.append(f"fair_swap_{pair}")
                available.append(f"fake_swap_{pair}")
                available.append(f"secret_swap_{pair}")
        else:
            # Cards are face-up: can only hide
            available.append("hide_cards")

        return available

    def step(self, action: str) -> Tuple[EpistemicState, RMState, float, bool, str]:
        """Execute an action and return (state, rm_state, reward, done, info)."""
        if self.done:
            raise RuntimeError("Episode is done — call reset()")

        self.step_count += 1

        # Apply action effects
        self.state = self._apply_action(action)

        # Step the reward machine
        reward, _, rm_info = self.reward_machine.step(self.state)

        # Check step limit
        self.done = self.step_count >= self.max_steps

        return self.state, self.reward_machine.get_state(), reward, self.done, rm_info

    def _apply_action(self, action: str) -> EpistemicState:
        """Apply physical and epistemic effects of an action."""
        s = self.state

        if action == "show_cards":
            return self._do_show_cards(s)
        elif action == "hide_cards":
            return self._do_hide_cards(s)
        elif action.startswith("fair_swap_"):
            pair = action.split("_")[-1]
            return self._do_fair_swap(s, pair)
        elif action.startswith("fake_swap_"):
            pair = action.split("_")[-1]
            return self._do_fake_swap(s, pair)
        elif action.startswith("secret_swap_"):
            pair = action.split("_")[-1]
            return self._do_secret_swap(s, pair)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _do_show_cards(self, s: EpistemicState) -> EpistemicState:
        """Show cards face-up — honest reveal of all positions.

        Physical: face_up = True, cards_moving = False
        Epistemic: Observer sees truth for all positions.
        """
        add = {"face_up"}
        remove = {"cards_moving"}

        # Observer sees the actual positions (honest reveal)
        for pos in ("money_L", "money_C", "money_R"):
            if s.has(pos):
                add.add(f"eb_{pos}")
                remove.add(f"eb_not_{pos}")
            else:
                add.add(f"eb_not_{pos}")
                remove.add(f"eb_{pos}")

        return s.set_fluents(add=add, remove=remove)

    def _do_hide_cards(self, s: EpistemicState) -> EpistemicState:
        """Hide cards face-down.

        Physical: face_up = False
        Epistemic: No belief change (observer remembers last seen).
        """
        return s.set_fluents(remove={"face_up"})

    def _do_fair_swap(self, s: EpistemicState, pair: str) -> EpistemicState:
        """Fair swap — both physical and belief positions swap.

        The observer sees the swap and tracks it correctly.
        """
        p1, p2 = SWAP_PAIRS[pair]
        add: Set[str] = {"cards_moving"}
        remove: Set[str] = set()

        # Physical swap
        has_p1 = s.has(p1)
        has_p2 = s.has(p2)
        if has_p1 and not has_p2:
            remove.add(p1)
            add.add(p2)
        elif has_p2 and not has_p1:
            remove.add(p2)
            add.add(p1)
        # If both or neither: no physical change (shouldn't happen with XOR)

        # Belief swap: observer tracks the swap
        eb_p1 = f"eb_{p1}"
        eb_p2 = f"eb_{p2}"
        eb_not_p1 = f"eb_not_{p1}"
        eb_not_p2 = f"eb_not_{p2}"

        has_eb_p1 = s.has(eb_p1)
        has_eb_p2 = s.has(eb_p2)
        has_eb_not_p1 = s.has(eb_not_p1)
        has_eb_not_p2 = s.has(eb_not_p2)

        # Swap beliefs for p1 and p2
        # After swap: belief about p1 becomes old belief about p2, and vice versa
        # Clear old beliefs first
        for f in (eb_p1, eb_p2, eb_not_p1, eb_not_p2):
            remove.add(f)

        # Set new beliefs (swapped)
        if has_eb_p2:
            add.add(eb_p1)
        if has_eb_not_p2:
            add.add(eb_not_p1)
        if has_eb_p1:
            add.add(eb_p2)
        if has_eb_not_p1:
            add.add(eb_not_p2)

        return s.set_fluents(add=add, remove=remove)

    def _do_fake_swap(self, s: EpistemicState, pair: str) -> EpistemicState:
        """Fake swap — observer thinks a swap happened, but cards don't move.

        Physical: NO change to money card positions.
        Epistemic: Observer updates beliefs as if a swap occurred.
        This is the primary deception mechanism.
        """
        p1, p2 = SWAP_PAIRS[pair]
        add: Set[str] = {"cards_moving"}
        remove: Set[str] = set()

        # NO physical swap!

        # Belief swap (same as fair swap): observer thinks it happened
        eb_p1 = f"eb_{p1}"
        eb_p2 = f"eb_{p2}"
        eb_not_p1 = f"eb_not_{p1}"
        eb_not_p2 = f"eb_not_{p2}"

        has_eb_p1 = s.has(eb_p1)
        has_eb_p2 = s.has(eb_p2)
        has_eb_not_p1 = s.has(eb_not_p1)
        has_eb_not_p2 = s.has(eb_not_p2)

        for f in (eb_p1, eb_p2, eb_not_p1, eb_not_p2):
            remove.add(f)

        if has_eb_p2:
            add.add(eb_p1)
        if has_eb_not_p2:
            add.add(eb_not_p1)
        if has_eb_p1:
            add.add(eb_p2)
        if has_eb_not_p1:
            add.add(eb_not_p2)

        return s.set_fluents(add=add, remove=remove)

    def _do_secret_swap(self, s: EpistemicState, pair: str) -> EpistemicState:
        """Secret swap — cards physically move, but observer doesn't notice.

        Physical: Money card positions swap.
        Epistemic: NO belief change (observer unaware).
        """
        p1, p2 = SWAP_PAIRS[pair]
        add: Set[str] = {"cards_moving"}
        remove: Set[str] = set()

        # Physical swap
        has_p1 = s.has(p1)
        has_p2 = s.has(p2)
        if has_p1 and not has_p2:
            remove.add(p1)
            add.add(p2)
        elif has_p2 and not has_p1:
            remove.add(p2)
            add.add(p1)

        # NO belief change!

        return s.set_fluents(add=add, remove=remove)

    # -------------------------------------------------------------------
    # Observation methods
    # -------------------------------------------------------------------

    def get_env_observation(self) -> Tuple[tuple, tuple]:
        """Return (physical_tuple, belief_tuple) — no RM state.

        Physical (5 bools): money_L, money_C, money_R, face_up, cards_moving
        Beliefs (6 bools): eb_money_L, eb_not_money_L, ..., eb_money_R, eb_not_money_R
        """
        s = self.state
        phys = (
            s.has("money_L"),
            s.has("money_C"),
            s.has("money_R"),
            s.has("face_up"),
            s.has("cards_moving"),
        )
        beliefs = (
            s.has("eb_money_L"),
            s.has("eb_not_money_L"),
            s.has("eb_money_C"),
            s.has("eb_not_money_C"),
            s.has("eb_money_R"),
            s.has("eb_not_money_R"),
        )
        return phys, beliefs

    def get_observation(self) -> Tuple[int, tuple, tuple]:
        """Return full observation (rm_index, phys_tuple, beliefs_tuple)."""
        phys, beliefs = self.get_env_observation()
        return self.reward_machine.get_state_index(), phys, beliefs

    def get_compact_state_str(self) -> str:
        """Human-readable state string for debugging."""
        s = self.state
        money = "?"
        for pos in ("L", "C", "R"):
            if s.has(f"money_{pos}"):
                money = pos
        face = "up" if s.has("face_up") else "down"
        rm = self.reward_machine.get_state().value
        return f"Money={money} face={face} RM={rm}"
