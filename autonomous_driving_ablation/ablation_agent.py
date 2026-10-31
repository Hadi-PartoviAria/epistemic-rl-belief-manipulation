"""Configurable Q-learning agent for the BEQRM ablation study (7x7 Waymo grid).

A single tabular Q-learning agent whose observation space is built from four
toggleable ingredients:

    use_explicit  -> include the 6-bit explicit-belief tuple (delta)
    use_implicit  -> include the 7-bit SAT-derived b_impl features
    use_rm        -> include the Reward Machine state index in the obs
    use_pbrs      -> add Ng/Harada/Russell potential-based reward shaping

Five ablation variants are defined below:

    Variant              phys  explicit  implicit  RM   PBRS
    qrm                   X       .         .       X    .
    beqrm_no_impl         X       X         .       X    .
    beqrm_no_explicit     X       .         X       X    .
    beqrm                 X       X         X       X    .
    beqrm_pbrs            X       X         X       X    X

The 7x7 physical observation tuple (10 elements) does NOT include a SAT-derived
verdict as the trailing element (unlike the old 4x4 env), so no stripping is
needed for no-belief variants.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from autonomous_driving_env import AutonomousDrivingEnv, GOAL, GRID_SIZE, ALL_ACTIONS
from reward_machine import RMState, RewardMachine


Action = str
Obs = tuple

# Potentials for RM states in the 7x7 Waymo RM (U0..U3, UT, UF).
RM_POTENTIALS: Dict[RMState, float] = {
    RMState.U0: 0.0,
    RMState.U1: 0.3,
    RMState.U2: 0.5,
    RMState.U3: 0.7,
    RMState.UT: 0.0,
    RMState.UF: 0.0,
}

POS_POTENTIAL_SCALE = 1.0
MAX_DIST = 2 * (GRID_SIZE - 1)


def _pos_potential(phys_obs: tuple) -> float:
    x, y = phys_obs[0], phys_obs[1]
    gx, gy = GOAL
    dist = abs(x - gx) + abs(y - gy)
    return POS_POTENTIAL_SCALE * (1.0 - dist / MAX_DIST)


def _argmax(qrow: Dict[Action, float], allowed: List[Action], q_init: float) -> Action:
    qmax = max(qrow.get(a, q_init) for a in allowed)
    best = [a for a in allowed if qrow.get(a, q_init) == qmax]
    return random.choice(best) if best else random.choice(allowed)


class AblationAgent:
    """Tabular Q-learner with ablatable observation, RM, and PBRS."""

    def __init__(
        self,
        reward_machine: RewardMachine,
        use_explicit: bool,
        use_implicit: bool,
        use_rm: bool,
        use_pbrs: bool,
        discount_factor: float = 0.99,
        learning_rate: float = 0.2,
        epsilon: float = 0.15,
        q_init: float = 0.0,
        all_actions: Optional[List[Action]] = None,
    ):
        self.rm = reward_machine
        self.use_explicit = use_explicit
        self.use_implicit = use_implicit
        self.use_rm = use_rm
        self.use_pbrs = use_pbrs

        self.gamma = discount_factor
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_init = q_init
        self.all_actions = all_actions or list(ALL_ACTIONS)

        self.Q: Dict[Obs, Dict[Action, float]] = {}
        self.training_episodes: int = 0

        self.sat_query_count: int = 0
        self.sat_query_time_s: float = 0.0

    def make_obs(self, env: AutonomousDrivingEnv, rm_state: RMState) -> Obs:
        phys, beliefs = env.get_env_observation()
        if not (self.use_explicit or self.use_implicit):
            # QRM variants: strip traffic-count evidence bins (indices 4-6).
            # Both PUDDLE and FLOODED share the same visual signal; without
            # traffic counts the agent sees only (x, y, heading, visual,
            # safe, legal, reached_B) and cannot distinguish them per episode.
            x, y, hdg, vis, _, _, _, safe, legal, reached_b = phys
            phys = (x, y, hdg, vis, safe, legal, reached_b)
        parts: List = [phys]
        if self.use_explicit:
            parts.append(beliefs)
        if self.use_implicit:
            t0 = time.perf_counter()
            b_impl = env.get_implicit_belief_features()
            self.sat_query_time_s += time.perf_counter() - t0
            self.sat_query_count += 1
            parts.append(b_impl)
        if self.use_rm:
            parts.append(rm_state.value)
        return tuple(parts)

    def _row(self, obs: Obs) -> Dict[Action, float]:
        if obs not in self.Q:
            self.Q[obs] = {a: self.q_init for a in self.all_actions}
        return self.Q[obs]

    def select_action(self, obs: Obs, available: List[Action]) -> Action:
        row = self._row(obs)
        if random.random() < self.epsilon:
            return random.choice(available)
        return _argmax(row, available, self.q_init)

    def get_policy_action(self, obs: Obs, available: List[Action]) -> Action:
        row = self._row(obs)
        return _argmax(row, available, self.q_init)

    def _potential(self, obs: Obs, rm_state: RMState) -> float:
        if not self.use_pbrs:
            return 0.0
        phi = _pos_potential(obs[0])
        if self.use_rm:
            phi += RM_POTENTIALS.get(rm_state, 0.0)
        return phi

    def update(
        self,
        old_obs: Obs,
        action: Action,
        new_obs: Obs,
        old_rm: RMState,
        new_rm: RMState,
        reward: float,
        new_available: List[Action],
    ) -> None:
        if self.use_pbrs:
            phi_old = self._potential(old_obs, old_rm)
            phi_new = self._potential(new_obs, new_rm)
            reward = reward + self.gamma * phi_new - phi_old

        row_old = self._row(old_obs)
        row_new = self._row(new_obs)
        qmax_next = max((row_new.get(a, self.q_init) for a in new_available),
                        default=0.0)
        delta = reward + self.gamma * qmax_next - row_old.get(action, self.q_init)
        row_old[action] = row_old.get(action, self.q_init) + self.lr * delta

    @property
    def num_states(self) -> int:
        return len(self.Q)


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

VARIANTS: Dict[str, Dict] = {
    "qrm":                dict(use_explicit=False, use_implicit=False,
                               use_rm=True,  use_pbrs=False),
    "beqrm_no_impl":      dict(use_explicit=True,  use_implicit=False,
                               use_rm=True,  use_pbrs=False),
    "beqrm_no_explicit":  dict(use_explicit=False, use_implicit=True,
                               use_rm=True,  use_pbrs=False),
    "beqrm":              dict(use_explicit=True,  use_implicit=True,
                               use_rm=True,  use_pbrs=False),
    "beqrm_pbrs":         dict(use_explicit=True,  use_implicit=True,
                               use_rm=True,  use_pbrs=True),
}

LABELS: Dict[str, str] = {
    "qrm":                "QRM (no beliefs)",
    "beqrm_no_impl":      "BEQRM – implicit",
    "beqrm_no_explicit":  "BEQRM – explicit",
    "beqrm":              "BEQRM",
    "beqrm_pbrs":         "BEQRM+PBRS",
}

COLORS: Dict[str, str] = {
    "qrm":                "#03A9F4",
    "beqrm_no_impl":      "#FF9800",
    "beqrm_no_explicit":  "#4CAF50",
    "beqrm":              "#1A237E",
    "beqrm_pbrs":         "#9C27B0",
}

# Central paper figure: the 4 most informative curves
CENTRAL_FIGURE_VARIANTS: List[str] = [
    "qrm",
    "beqrm",
    "beqrm_pbrs",
]
