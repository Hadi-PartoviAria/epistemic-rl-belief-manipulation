"""
Chop Cup Environment — Belief-based RL environment for the magic trick.

FAITHFULLY implements the action effects from magic_cup.json.
Every fluent update is a direct translation of the JSON specification.

Models the Chop Cup with:
  - 1 cup with a magnet
  - 1 magnetic ball (hidden, stuck to cup)
  - 1 regular ball (can be palmed, placed, revealed)
  - optional limes (for the finale)

State is composed of:
  - Physical fluents: actual world state
  - Explicit beliefs (Δ): what the Observer directly observed
  - Reward Machine state: tracked by separate RM object, mirroring rm_uX fluents

Implicit beliefs (□) are computed on-demand via the epistemic logic engine.

Architecture mirrors the Rust `bbmdp.rs` implementation:
  - Actions have guards (preconditions) and effects (state transitions)
  - Effects update both physical fluents AND explicit beliefs
  - The SAT solver derives implicit beliefs for evaluation/analysis
  - RM transitions are encoded in action effects (matching JSON exactly)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from epistemic_logic import CommonGround, EpistemicState
from reward_machine import RewardMachine, RMState


# ---------------------------------------------------------------------------
# Physical fluent names (matching magic_cup.json)
# ---------------------------------------------------------------------------
MAG_STUCK = "mag_stuck"        # magnetic ball stuck inside cup
MAG_VISIBLE = "mag_visible"    # magnetic ball visible (cup lifted)
REG_UNDER = "reg_under"        # regular ball under the cup
REG_TABLE = "reg_table"        # regular ball on the table
REG_HAND = "reg_hand"          # regular ball palmed in hand
CUP_DOWN = "cup_down"          # cup is covering its contents
BALL_VISIBLE = "ball_visible"  # a ball is visible under cup (derived)
LIME_CUP = "lime_cup"          # lime is under the cup
LIME_VISIBLE = "lime_visible"  # lime is visible (cup lifted with lime)

# Explicit belief fluent names (Δ) — matching JSON's △ notation
EB_REG_UNDER = "eb_reg_under"          # △(reg_under)
EB_NOT_REG_UNDER = "eb_not_reg_under"  # △(¬reg_under)
EB_REG_TABLE = "eb_reg_table"          # △(reg_table)
EB_NOT_REG_TABLE = "eb_not_reg_table"  # △(¬reg_table)
EB_REG_HAND = "eb_reg_hand"            # △(reg_hand)
EB_NOT_REG_HAND = "eb_not_reg_hand"    # △(¬reg_hand)

# Lime explicit belief fluents (for lime sub-routine)
EB_LIME_CUP = "eb_lime_cup"            # △(lime_cup) — observer saw lime
EB_NOT_LIME_CUP = "eb_not_lime_cup"    # △(¬lime_cup) — observer believes no lime

# Observer's model variables (for Common Ground SAT solver)
OBSERVER_VARS = ["reg_under", "reg_table", "reg_hand"]


# ---------------------------------------------------------------------------
# Chop Cup Environment
# ---------------------------------------------------------------------------

class ChopCupEnv:
    """Chop Cup magic trick environment with epistemic state.

    This is a belief-based MDP where:
      - The agent (Magician) has full knowledge of the physical state
      - The Observer has partial knowledge via explicit/implicit beliefs
      - Success = creating epistemic divergence (□φ ∧ ¬φ)

    Action effects faithfully match magic_cup.json.
    RM transitions are condition-based (evaluated against epistemic state).
    """

    def __init__(
        self,
        enable_limes: bool = False,
        max_limes: int = 5,
        max_steps: int = 50,
        shaping: bool = False,
    ):
        self.enable_limes = enable_limes
        self.max_limes = max_limes
        self.max_steps = max_steps

        # Common Ground: regular ball is in exactly one of {under, table, hand}
        # Observer does NOT know about the magnetic ball or limes
        self.common_ground = CommonGround(
            variables=OBSERVER_VARS,
            exactly_one_groups=[OBSERVER_VARS],
        )

        self.reward_machine = RewardMachine(enable_limes=enable_limes, shaping=shaping)
        self.state: Optional[EpistemicState] = None
        self.lime_count: int = 0
        self.step_count: int = 0
        self.done: bool = False

    def reset(self) -> Tuple[EpistemicState, RMState]:
        """Reset to initial state matching JSON constraint exactly.

        JSON initial_states.constraint:
          mag_stuck & !mag_visible & reg_table & !reg_under & !reg_hand & cup_down
          & △reg_table & !△!reg_table & !△reg_under & !△!reg_under
          & !△reg_hand & !△!reg_hand & rm_u0 & !rm_u1 & !rm_u2 & !rm_u3
        """
        self.reward_machine.reset()
        self.lime_count = 0
        self.step_count = 0
        self.done = False

        initial_fluents = frozenset({
            MAG_STUCK,          # mag_stuck = true
            REG_TABLE,          # reg_table = true
            CUP_DOWN,           # cup_down = true
            EB_REG_TABLE,       # △reg_table = true
            # All other fluents are false (not in the set)
        })

        # When limes enabled, observer starts believing no lime
        if self.enable_limes:
            initial_fluents = initial_fluents | frozenset({EB_NOT_LIME_CUP})

        self.state = EpistemicState(initial_fluents, self.common_ground)
        return self.state, self.reward_machine.get_state()

    def get_available_actions(self) -> List[str]:
        """Return actions whose guards are satisfied.

        Guards from magic_cup.json:
          lift_cup:    guard = cup_down
          place_cup:   guard = !(cup_down)
          palm_ball:   guard = reg_table
          reveal_hand: no guard (always available)
        """
        if self.state is None or self.done:
            return []

        actions = []
        s = self.state

        # lift_cup: guard = cup_down
        if s.has(CUP_DOWN):
            actions.append("lift_cup")

        # place_cup: guard = !(cup_down)
        if not s.has(CUP_DOWN):
            actions.append("place_cup")

        # palm_ball: guard = reg_table
        if s.has(REG_TABLE):
            actions.append("palm_ball")

        # reveal_hand: no guard
        actions.append("reveal_hand")

        # Extended actions (load_lime) — not in base JSON
        # Available only during lime phase (U4) when cup is down
        if self.enable_limes:
            rm = self.reward_machine.get_state()
            if s.has(CUP_DOWN) and self.lime_count < self.max_limes:
                if rm == RMState.U4:
                    actions.append("load_lime")

        return actions

    def step(self, action: str) -> Tuple[EpistemicState, RMState, float, bool, str]:
        """Execute an action and return (new_state, rm_state, reward, done, info).

        1. Apply action effects on physical fluents and explicit beliefs (matching JSON)
        2. Apply RM transition for this action (matching JSON rm_uX effects)
        3. Return updated state
        """
        assert self.state is not None, "Must call reset() first"
        assert not self.done, "Episode already finished"
        assert action in self.get_available_actions(), f"Invalid action: {action}"

        self.step_count += 1
        info = ""

        # Apply action effects on physical + belief fluents
        if action == "lift_cup":
            self.state, info = self._do_lift_cup()
        elif action == "place_cup":
            self.state, info = self._do_place_cup()
        elif action == "palm_ball":
            self.state, info = self._do_palm_ball()
        elif action == "reveal_hand":
            self.state, info = self._do_reveal_hand()
        elif action == "load_lime":
            self.state, info = self._do_load_lime()

        # Apply RM transition (condition-based: RM evaluates epistemic state)
        rm_reward, rm_terminal, rm_info = self.reward_machine.step(self.state)
        if rm_info:
            info += f" | RM: {rm_info}"

        # Check termination: only by step limit, not RM state
        self.done = self.step_count >= self.max_steps

        return self.state, self.reward_machine.get_state(), rm_reward, self.done, info

    # -------------------------------------------------------------------
    # Action implementations — faithfully mirroring magic_cup.json effects
    # -------------------------------------------------------------------

    def _do_lift_cup(self) -> Tuple[EpistemicState, str]:
        """lift_cup — Lift the cup to reveal what's underneath.

        JSON effects (each line: fluent := formula_over_old_state):
          mag_stuck     := ⊥
          mag_visible   := (mag_visible) | (mag_stuck)
          reg_under     := ⊥
          reg_table     := (reg_table) | (reg_under)
          reg_hand      := reg_hand                    [unchanged]
          cup_down      := ⊥
          △reg_under    := ⊥
          △!reg_under   := T
          △reg_table    := (△reg_table) | (reg_under)  [NOTE: depends on physical reg_under]
          △!reg_table   := (△!reg_table) & !(reg_under)
          △reg_hand     := △reg_hand                   [unchanged]
          △!reg_hand    := △!reg_hand                  [unchanged]

        RM: condition h(ball_visible) may fire u2→u3
        """
        s = self.state
        old_mag_stuck = s.has(MAG_STUCK)
        old_mag_visible = s.has(MAG_VISIBLE)
        old_reg_under = s.has(REG_UNDER)
        old_reg_table = s.has(REG_TABLE)
        old_reg_hand = s.has(REG_HAND)
        old_eb_reg_table = s.has(EB_REG_TABLE)
        old_eb_not_reg_table = s.has(EB_NOT_REG_TABLE)
        old_eb_reg_hand = s.has(EB_REG_HAND)
        old_eb_not_reg_hand = s.has(EB_NOT_REG_HAND)

        # Compute new fluent values from JSON formulas
        new_fluents: Set[str] = set()

        # mag_stuck := ⊥  (always false after lift)
        # mag_visible := mag_visible | mag_stuck
        if old_mag_visible or old_mag_stuck:
            new_fluents.add(MAG_VISIBLE)

        # reg_under := ⊥  (nothing under cup after lift)
        # reg_table := reg_table | reg_under
        if old_reg_table or old_reg_under:
            new_fluents.add(REG_TABLE)

        # reg_hand := reg_hand (unchanged)
        if old_reg_hand:
            new_fluents.add(REG_HAND)

        # cup_down := ⊥

        # △reg_under := ⊥  (observer sees nothing under cup)
        # △!reg_under := T  (observer knows nothing under cup)
        new_fluents.add(EB_NOT_REG_UNDER)

        # △reg_table := △reg_table | reg_under  (if ball was under, observer now sees it on table)
        if old_eb_reg_table or old_reg_under:
            new_fluents.add(EB_REG_TABLE)

        # △!reg_table := △!reg_table & !reg_under  (only keep "not on table" if ball wasn't under)
        if old_eb_not_reg_table and not old_reg_under:
            new_fluents.add(EB_NOT_REG_TABLE)

        # △reg_hand := △reg_hand (unchanged)
        if old_eb_reg_hand:
            new_fluents.add(EB_REG_HAND)

        # △!reg_hand := △!reg_hand (unchanged)
        if old_eb_not_reg_hand:
            new_fluents.add(EB_NOT_REG_HAND)

        # Derived fluent: ball_visible (not in JSON, but useful for analysis)
        if old_mag_stuck or old_reg_under:
            new_fluents.add(BALL_VISIBLE)

        # Limes: physical state
        if s.has(LIME_CUP):
            new_fluents.add(LIME_CUP)
            new_fluents.add(LIME_VISIBLE)
            # Observer SEES the lime → update lime beliefs
            new_fluents.add(EB_LIME_CUP)
            # EB_NOT_LIME_CUP is NOT added (cleared = observer no longer believes no lime)
        else:
            # No lime — preserve existing lime beliefs unchanged
            if s.has(EB_LIME_CUP):
                new_fluents.add(EB_LIME_CUP)
            if s.has(EB_NOT_LIME_CUP):
                new_fluents.add(EB_NOT_LIME_CUP)

        info = f"lift_cup: mag_stuck={old_mag_stuck}, reg_under={old_reg_under}"
        return EpistemicState(frozenset(new_fluents), self.common_ground), info

    def _do_place_cup(self) -> Tuple[EpistemicState, str]:
        """place_cup — Place the cup back down.

        JSON effects:
          mag_stuck     := mag_stuck                   [unchanged]
          mag_visible   := mag_visible                 [unchanged]
          reg_under     := reg_table                   [ball on table goes under cup]
          reg_table     := ⊥
          reg_hand      := reg_hand                    [unchanged]
          cup_down      := T
          △reg_under    := △reg_table                  [belief transfers: table → under]
          △!reg_under   := △!reg_table
          △reg_table    := ⊥
          △!reg_table   := T
          △reg_hand     := △reg_hand                   [unchanged]
          △!reg_hand    := △!reg_hand                  [unchanged]

        RM: condition c(cup_down) may fire u3→u0
        """
        s = self.state
        old_mag_stuck = s.has(MAG_STUCK)
        old_mag_visible = s.has(MAG_VISIBLE)
        old_reg_table = s.has(REG_TABLE)
        old_reg_hand = s.has(REG_HAND)
        old_eb_reg_table = s.has(EB_REG_TABLE)
        old_eb_not_reg_table = s.has(EB_NOT_REG_TABLE)
        old_eb_reg_hand = s.has(EB_REG_HAND)
        old_eb_not_reg_hand = s.has(EB_NOT_REG_HAND)

        new_fluents: Set[str] = set()

        # mag_stuck := mag_stuck
        if old_mag_stuck:
            new_fluents.add(MAG_STUCK)
        # mag_visible := mag_visible
        if old_mag_visible:
            new_fluents.add(MAG_VISIBLE)

        # reg_under := reg_table  (ball on table goes under cup)
        if old_reg_table:
            new_fluents.add(REG_UNDER)
        # reg_table := ⊥

        # reg_hand := reg_hand
        if old_reg_hand:
            new_fluents.add(REG_HAND)

        # cup_down := T
        new_fluents.add(CUP_DOWN)

        # △reg_under := △reg_table  (BELIEF TRANSFER: what observer thought about table → under)
        if old_eb_reg_table:
            new_fluents.add(EB_REG_UNDER)
        # △!reg_under := △!reg_table
        if old_eb_not_reg_table:
            new_fluents.add(EB_NOT_REG_UNDER)
        # △reg_table := ⊥
        # △!reg_table := T
        new_fluents.add(EB_NOT_REG_TABLE)

        # △reg_hand := △reg_hand
        if old_eb_reg_hand:
            new_fluents.add(EB_REG_HAND)
        # △!reg_hand := △!reg_hand
        if old_eb_not_reg_hand:
            new_fluents.add(EB_NOT_REG_HAND)

        # ball_visible := false (cup covers), lime_visible := false
        # Lime handling: when placing cup, lime is "removed" (audience sees it taken out)
        # LIME_CUP and LIME_VISIBLE are both cleared (not added to new_fluents)
        # Observer now believes no lime under cup
        if s.has(LIME_VISIBLE) or s.has(LIME_CUP):
            # Lime was present — now removed as part of the show
            new_fluents.add(EB_NOT_LIME_CUP)
            # EB_LIME_CUP NOT added (cleared)
        else:
            # No lime involved — preserve existing lime beliefs
            if s.has(EB_LIME_CUP):
                new_fluents.add(EB_LIME_CUP)
            if s.has(EB_NOT_LIME_CUP):
                new_fluents.add(EB_NOT_LIME_CUP)

        info = f"place_cup: reg_table→under={old_reg_table}"
        return EpistemicState(frozenset(new_fluents), self.common_ground), info

    def _do_palm_ball(self) -> Tuple[EpistemicState, str]:
        """palm_ball — THE DECEPTIVE ACTION. Secret ball theft.

        JSON effects:
          mag_stuck     := mag_stuck                   [unchanged]
          mag_visible   := mag_visible                 [unchanged]
          reg_under     := reg_under                   [unchanged]
          reg_table     := ⊥                           [ball leaves table]
          reg_hand      := reg_table                   [ball goes to hand]
          cup_down      := cup_down                    [unchanged]
          △reg_under    := △reg_table                  [BELIEF TRANSFER: table → under]
          △!reg_under   := △!reg_table                 [BELIEF TRANSFER]
          △reg_table    := ⊥
          △!reg_table   := T
          △reg_hand     := ⊥                           [observer doesn't see ball in hand]
          △!reg_hand    := T                           [observer believes hand is empty]

        RM: condition a(eb_reg_under) may fire u0→u1

        The belief transfer △reg_under := △reg_table is CRITICAL:
        - Before palm: △reg_table = true (observer sees ball on table)
        - After palm: △reg_under = true (observer "transfers" that belief to under)
        - Combined with △!reg_hand = true and △!reg_table = true
        - The SAT solver ALSO deduces □(reg_under) from the Common Ground

        Both the explicit belief AND the implicit deduction agree: the Observer
        thinks the ball went under the cup. But it's actually in the hand!
        """
        s = self.state
        old_mag_stuck = s.has(MAG_STUCK)
        old_mag_visible = s.has(MAG_VISIBLE)
        old_reg_under = s.has(REG_UNDER)
        old_reg_table = s.has(REG_TABLE)
        old_cup_down = s.has(CUP_DOWN)
        old_eb_reg_table = s.has(EB_REG_TABLE)
        old_eb_not_reg_table = s.has(EB_NOT_REG_TABLE)

        new_fluents: Set[str] = set()

        # mag_stuck := mag_stuck
        if old_mag_stuck:
            new_fluents.add(MAG_STUCK)
        # mag_visible := mag_visible
        if old_mag_visible:
            new_fluents.add(MAG_VISIBLE)

        # reg_under := reg_under (unchanged)
        if old_reg_under:
            new_fluents.add(REG_UNDER)
        # reg_table := ⊥ (ball no longer on table)
        # reg_hand := reg_table (ball goes from table to hand)
        if old_reg_table:
            new_fluents.add(REG_HAND)

        # cup_down := cup_down (unchanged)
        if old_cup_down:
            new_fluents.add(CUP_DOWN)

        # △reg_under := △reg_table  *** THE KEY BELIEF TRANSFER ***
        if old_eb_reg_table:
            new_fluents.add(EB_REG_UNDER)
        # △!reg_under := △!reg_table
        if old_eb_not_reg_table:
            new_fluents.add(EB_NOT_REG_UNDER)
        # △reg_table := ⊥
        # △!reg_table := T
        new_fluents.add(EB_NOT_REG_TABLE)
        # △reg_hand := ⊥
        # △!reg_hand := T
        new_fluents.add(EB_NOT_REG_HAND)

        # Derived: ball_visible unchanged
        if s.has(BALL_VISIBLE):
            new_fluents.add(BALL_VISIBLE)
        # Limes (physical) unchanged
        if s.has(LIME_CUP):
            new_fluents.add(LIME_CUP)
        if s.has(LIME_VISIBLE):
            new_fluents.add(LIME_VISIBLE)
        # Lime beliefs unchanged
        if s.has(EB_LIME_CUP):
            new_fluents.add(EB_LIME_CUP)
        if s.has(EB_NOT_LIME_CUP):
            new_fluents.add(EB_NOT_LIME_CUP)

        info = "palm_ball: DECEPTION — ball to hand, △reg_under := △reg_table"
        return EpistemicState(frozenset(new_fluents), self.common_ground), info

    def _do_reveal_hand(self) -> Tuple[EpistemicState, str]:
        """reveal_hand — Show what's in the magician's hand.

        JSON effects:
          mag_stuck     := mag_stuck                   [unchanged]
          mag_visible   := mag_visible                 [unchanged]
          reg_under     := reg_under                   [unchanged]
          reg_table     := reg_table                   [unchanged]
          reg_hand      := reg_hand                    [unchanged]
          cup_down      := cup_down                    [unchanged]
          △reg_under    := △reg_under                  [unchanged]
          △!reg_under   := △!reg_under                 [unchanged]
          △reg_table    := △reg_table                  [unchanged]
          △!reg_table   := △!reg_table                 [unchanged]
          △reg_hand     := reg_hand                    [reveal truth about hand]
          △!reg_hand    := !(reg_hand)                 [reveal truth about hand]

        RM: condition b(□reg_hand) may fire u1→u2 if hand revealed with ball
        """
        s = self.state
        has_ball = s.has(REG_HAND)

        # Only △reg_hand and △!reg_hand change
        add: Set[str] = set()
        remove: Set[str] = set()

        # △reg_hand := reg_hand (physical truth)
        if has_ball:
            add.add(EB_REG_HAND)
            remove.add(EB_NOT_REG_HAND)
        else:
            # △!reg_hand := !(reg_hand) = true
            remove.add(EB_REG_HAND)
            add.add(EB_NOT_REG_HAND)

        info = f"reveal_hand: ball_in_hand={has_ball}"
        return s.set_fluents(add=add, remove=remove), info

    def _do_load_lime(self) -> Tuple[EpistemicState, str]:
        """Secretly load a lime under the cup (not in base JSON)."""
        add: Set[str] = set()
        add.add(LIME_CUP)
        self.lime_count += 1
        info = f"load_lime: count={self.lime_count}"
        return self.state.set_fluents(add=add), info

    # -------------------------------------------------------------------
    # State representation for the RL agent
    # -------------------------------------------------------------------

    def get_env_observation(self) -> tuple:
        """Return a hashable observation WITHOUT RM state (for QRM agent).

        QRM maintains separate Q-tables per RM state, so the
        observation should only contain environmental state.
        """
        if self.state is None:
            return (0,)

        phys = (
            self.state.has(MAG_STUCK),
            self.state.has(MAG_VISIBLE),
            self.state.has(REG_UNDER),
            self.state.has(REG_TABLE),
            self.state.has(REG_HAND),
            self.state.has(CUP_DOWN),
            self.state.has(BALL_VISIBLE),
            self.state.has(LIME_CUP),
            self.state.has(LIME_VISIBLE),
            self.lime_count,
        )

        beliefs = (
            self.state.has(EB_REG_UNDER),
            self.state.has(EB_NOT_REG_UNDER),
            self.state.has(EB_REG_TABLE),
            self.state.has(EB_NOT_REG_TABLE),
            self.state.has(EB_REG_HAND),
            self.state.has(EB_NOT_REG_HAND),
            self.state.has(EB_LIME_CUP),
            self.state.has(EB_NOT_LIME_CUP),
        )

        return (phys, beliefs)

    def get_observation(self) -> tuple:
        """Return a hashable observation for the Q-table.

        The agent (Magician) has FULL observability — it knows both
        the physical state AND the Observer's beliefs.

        Returns a tuple of (rm_state_index, physical_key, belief_key).
        """
        if self.state is None:
            return (0,)

        rm_idx = self.reward_machine.get_state_index()

        # Physical state key
        phys = (
            self.state.has(MAG_STUCK),
            self.state.has(MAG_VISIBLE),
            self.state.has(REG_UNDER),
            self.state.has(REG_TABLE),
            self.state.has(REG_HAND),
            self.state.has(CUP_DOWN),
            self.state.has(BALL_VISIBLE),
            self.state.has(LIME_CUP),
            self.state.has(LIME_VISIBLE),
            self.lime_count,
        )

        # Belief state key
        beliefs = (
            self.state.has(EB_REG_UNDER),
            self.state.has(EB_NOT_REG_UNDER),
            self.state.has(EB_REG_TABLE),
            self.state.has(EB_NOT_REG_TABLE),
            self.state.has(EB_REG_HAND),
            self.state.has(EB_NOT_REG_HAND),
            self.state.has(EB_LIME_CUP),
            self.state.has(EB_NOT_LIME_CUP),
        )

        return (rm_idx, phys, beliefs)

    def get_compact_state_str(self) -> str:
        """Human-readable state summary."""
        if self.state is None:
            return "None"
        s = self.state
        parts = []
        if s.has(MAG_STUCK): parts.append("mag:stuck")
        if s.has(MAG_VISIBLE): parts.append("mag:vis")
        if s.has(REG_UNDER): parts.append("reg:under")
        if s.has(REG_TABLE): parts.append("reg:table")
        if s.has(REG_HAND): parts.append("reg:HAND")
        if s.has(CUP_DOWN): parts.append("cup:DOWN")
        else: parts.append("cup:UP")
        if s.has(BALL_VISIBLE): parts.append("ball:VIS")
        if self.lime_count > 0: parts.append(f"limes:{self.lime_count}")
        parts.append(f"RM:{self.reward_machine.get_state().value}")

        # Show key explicit beliefs
        eb = []
        if s.has(EB_REG_UNDER): eb.append("△(under)")
        if s.has(EB_NOT_REG_UNDER): eb.append("△(¬under)")
        if s.has(EB_REG_TABLE): eb.append("△(table)")
        if s.has(EB_NOT_REG_TABLE): eb.append("△(¬table)")
        if s.has(EB_REG_HAND): eb.append("△(hand)")
        if s.has(EB_NOT_REG_HAND): eb.append("△(¬hand)")
        if eb:
            parts.append("Δ={" + ",".join(eb) + "}")

        # Show key implicit beliefs
        ib = []
        if s.implicit_belief("reg_under", positive=True): ib.append("□(under)")
        if s.implicit_belief("reg_hand", positive=True): ib.append("□(hand)")
        if ib:
            parts.append(" ".join(ib))

        return " | ".join(parts)
