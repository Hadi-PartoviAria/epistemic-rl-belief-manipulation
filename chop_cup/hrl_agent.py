"""
HRL Baseline — Hierarchical RL with Options Framework (No Epistemic Reasoning).

Two-level hierarchy for the Chop Cup magic trick:
  Meta-controller: SMDP Q-learning to select subgoals (options)
  Controller:      Q-learning to achieve subgoals with primitive actions

Observation space: Physical state ONLY (10 features).
  Does NOT include explicit beliefs (Δ), implicit beliefs (□), or RM state.
  The agent cannot observe what the observer believes — it only sees
  the physical world (ball positions, cup state, lime state).

Subgoals (options):
  0. palm      — Ball in magician's hand (reg_hand = True)
  1. show_ball — Ball visible under cup (ball_visible = True)
  2. reset     — Cup down, nothing visible (cup_down ∧ ¬ball_visible ∧ ¬lime_visible)
  3. load_lime — Lime under cup (lime_cup = True)
  4. show_lime — Lime visible (lime_visible = True)

Comparison with HERL-QRM:
  - HRL uses options with hand-crafted physical subgoals
  - HERL-QRM uses a Reward Machine with epistemic conditions
  - HRL sees physical state only (10 features)
  - HERL-QRM sees physical + epistemic state (18 features + RM state)
  - HRL has no SAT-based implicit belief reasoning
  - HERL-QRM uses □φ via SAT entailment for RM transitions

Reference:
  Sutton, Precup, Singh (1999). "Between MDPs and Semi-MDPs:
  A Framework for Temporal Abstraction in RL." Artificial Intelligence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Subgoals (Options)
# ---------------------------------------------------------------------------

@dataclass
class Subgoal:
    """A subgoal/option in the hierarchical framework."""
    name: str
    check: Callable  # (EpistemicState) -> bool, uses physical fluents only
    max_steps: int = 15  # max primitive steps per option invocation


# Physical-state subgoals — NO epistemic conditions
SUBGOALS = [
    Subgoal("palm",      lambda s: s.has("reg_hand"),       15),
    Subgoal("show_ball", lambda s: s.has("ball_visible"),   15),
    Subgoal("reset",     lambda s: (s.has("cup_down")
                                    and not s.has("ball_visible")
                                    and not s.has("lime_visible")), 15),
    Subgoal("load_lime", lambda s: s.has("lime_cup"),       15),
    Subgoal("show_lime", lambda s: s.has("lime_visible"),   15),
]


# ---------------------------------------------------------------------------
# HRL Agent
# ---------------------------------------------------------------------------

class HRLAgent:
    """Hierarchical RL with Options — physical-only baseline.

    Meta-level:  SMDP Q-learning selects among subgoals (options).
    Low-level:   Standard Q-learning selects primitive actions to achieve
                 the current subgoal.

    Physical observation:
      (mag_stuck, mag_visible, reg_under, reg_table, reg_hand,
       cup_down, ball_visible, lime_cup, lime_visible, lime_count)
    """

    ALL_ACTIONS = ["lift_cup", "place_cup", "palm_ball", "reveal_hand", "load_lime"]

    def __init__(
        self,
        discount: float = 0.9,
        lr: float = 0.1,
        epsilon: float = 0.1,          # controller exploration
        meta_epsilon: float = 0.2,     # meta-controller exploration
        q_init: float = 0.0,
        intrinsic_bonus: float = 1.0,  # reward for achieving subgoal
        step_cost: float = 0.01,       # per-step penalty within option
    ):
        self.gamma = discount
        self.lr = lr
        self.epsilon = epsilon
        self.meta_epsilon = meta_epsilon
        self.q_init = q_init
        self.intrinsic_bonus = intrinsic_bonus
        self.step_cost = step_cost

        self.num_subgoals = len(SUBGOALS)

        # Meta Q-table:  meta_Q[phys_obs][subgoal_idx] → float
        self.meta_Q: Dict[tuple, Dict[int, float]] = {}

        # Controller Q-tables (one per subgoal):
        # ctrl_Q[sg_idx][phys_obs][action] → float
        self.ctrl_Q: List[Dict[tuple, Dict[str, float]]] = [
            {} for _ in range(self.num_subgoals)
        ]

        # Option tracking (training only)
        self._sg: Optional[int] = None          # current subgoal index
        self._opt_steps: int = 0                # steps in current option
        self._opt_start: Optional[tuple] = None  # option start observation
        self._opt_ext_r: float = 0.0            # cumulative extrinsic reward
        self._opt_gamma: float = 1.0            # γ^k for SMDP update

        # Stats
        self.training_episodes: int = 0
        self._step_count: int = 0
        self._transition_count: int = 0

    # --- Observation ---

    @staticmethod
    def get_physical_obs(env) -> tuple:
        """Extract physical-only observation (no beliefs, no RM state)."""
        s = env.state
        return (
            s.has("mag_stuck"),
            s.has("mag_visible"),
            s.has("reg_under"),
            s.has("reg_table"),
            s.has("reg_hand"),
            s.has("cup_down"),
            s.has("ball_visible"),
            s.has("lime_cup"),
            s.has("lime_visible"),
            env.lime_count,
        )

    # --- Q-table helpers ---

    def _init_meta(self, obs: tuple) -> None:
        if obs not in self.meta_Q:
            self.meta_Q[obs] = {i: self.q_init for i in range(self.num_subgoals)}

    def _init_ctrl(self, sg: int, obs: tuple) -> None:
        if obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][obs] = {a: self.q_init for a in self.ALL_ACTIONS}

    # --- Training: option lifecycle ---

    def begin_episode(self) -> None:
        """Reset option state at start of new episode."""
        self._sg = None
        self._opt_steps = 0
        self._opt_start = None
        self._opt_ext_r = 0.0
        self._opt_gamma = 1.0

    def needs_option(self) -> bool:
        """True if no option is currently active."""
        return self._sg is None

    def start_option(self, phys_obs: tuple) -> int:
        """Select and start a new option (ε-greedy meta-controller)."""
        self._init_meta(phys_obs)

        if random.random() < self.meta_epsilon:
            sg = random.randint(0, self.num_subgoals - 1)
        else:
            qvals = self.meta_Q[phys_obs]
            maxq = max(qvals.values())
            best = [i for i, q in qvals.items() if q == maxq]
            sg = random.choice(best)

        self._sg = sg
        self._opt_steps = 0
        self._opt_start = phys_obs
        self._opt_ext_r = 0.0
        self._opt_gamma = 1.0
        return sg

    def select_action(self, phys_obs: tuple, available: List[str]) -> str:
        """ε-greedy action selection within current option."""
        sg = self._sg
        self._init_ctrl(sg, phys_obs)

        if random.random() < self.epsilon:
            return random.choice(available)

        qvals = self.ctrl_Q[sg][phys_obs]
        maxq = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == maxq]
        return random.choice(best)

    def update(
        self,
        old_phys: tuple,
        action: str,
        new_phys: tuple,
        env_state,            # EpistemicState — for subgoal check (physical fluents)
        extrinsic_reward: float,
        available_new: List[str],
        episode_done: bool,
    ) -> bool:
        """Process one step and update Q-tables.

        Returns True if the current option terminated.
        """
        sg = self._sg
        self._opt_steps += 1
        self._step_count += 1
        self._transition_count += 1

        # Check option termination
        achieved = SUBGOALS[sg].check(env_state)
        timeout = self._opt_steps >= SUBGOALS[sg].max_steps
        option_done = achieved or timeout or episode_done

        # --- Controller Q-update (intrinsic reward) ---
        self._init_ctrl(sg, old_phys)
        self._init_ctrl(sg, new_phys)

        r_int = self.intrinsic_bonus if achieved else -self.step_cost
        old_q = self.ctrl_Q[sg][old_phys].get(action, self.q_init)

        if option_done:
            td = r_int - old_q  # terminal: no bootstrapping
        else:
            acts = available_new if available_new else self.ALL_ACTIONS
            maxq_next = max(self.ctrl_Q[sg][new_phys].get(a, self.q_init) for a in acts)
            td = r_int + self.gamma * maxq_next - old_q

        self.ctrl_Q[sg][old_phys][action] = old_q + self.lr * td

        # --- Accumulate extrinsic reward for meta-controller ---
        self._opt_ext_r += self._opt_gamma * extrinsic_reward
        self._opt_gamma *= self.gamma

        # --- Meta-controller SMDP Q-update (on option termination) ---
        if option_done:
            start = self._opt_start
            self._init_meta(start)
            self._init_meta(new_phys)

            k = self._opt_steps
            R = self._opt_ext_r

            old_meta_q = self.meta_Q[start][sg]
            if episode_done:
                meta_target = R
            else:
                meta_target = R + (self.gamma ** k) * max(self.meta_Q[new_phys].values())

            self.meta_Q[start][sg] = old_meta_q + self.lr * (meta_target - old_meta_q)
            self._sg = None  # option finished

        return option_done

    # --- Evaluation: greedy, read-only ---

    def select_option_greedy(self, phys_obs: tuple) -> int:
        """Greedy option selection (for evaluation)."""
        self._init_meta(phys_obs)
        qvals = self.meta_Q[phys_obs]
        maxq = max(qvals.values())
        best = [i for i, q in qvals.items() if q == maxq]
        return random.choice(best)

    def select_action_greedy(self, phys_obs: tuple, sg: int, available: List[str]) -> str:
        """Greedy action selection within option (for evaluation)."""
        self._init_ctrl(sg, phys_obs)
        qvals = self.ctrl_Q[sg][phys_obs]
        maxq = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == maxq]
        return random.choice(best)

    # --- Properties ---

    @property
    def num_states(self) -> int:
        total = len(self.meta_Q)
        for cq in self.ctrl_Q:
            total += len(cq)
        return total

    @property
    def num_transitions(self) -> int:
        return self._transition_count
