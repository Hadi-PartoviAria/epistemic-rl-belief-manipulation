"""HRL — Hierarchical RL Baseline for Three-Card Monte (Physical-Only).

Two-level hierarchy:
  Meta-controller: SMDP Q-learning over subgoals
  Controller: Per-subgoal Q-learning over primitive actions

Observation: Physical state only (5 features) — no beliefs, no RM state.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from three_card_env import ThreeCardMonteEnv, ALL_ACTIONS


# Subgoal definitions — physical conditions only
SUBGOALS = [
    {
        "name": "reveal",
        "check": lambda env: env.state.has("face_up"),
        "max_steps": 10,
    },
    {
        "name": "hide",
        "check": lambda env: not env.state.has("face_up"),
        "max_steps": 10,
    },
    {
        "name": "shuffle",
        "check": lambda env: env.state.has("cards_moving") and not env.state.has("face_up"),
        "max_steps": 10,
    },
]


class HRLAgent:
    """Hierarchical RL agent with physical-only observations."""

    def __init__(
        self,
        discount: float = 0.9,
        lr: float = 0.1,
        epsilon: float = 0.1,
        meta_epsilon: float = 0.2,
        q_init: float = 0.0,
        intrinsic_bonus: float = 1.0,
        step_cost: float = 0.01,
    ):
        self.gamma = discount
        self.lr = lr
        self.epsilon = epsilon
        self.meta_epsilon = meta_epsilon
        self.q_init = q_init
        self.intrinsic_bonus = intrinsic_bonus
        self.step_cost = step_cost
        self.all_actions = ALL_ACTIONS

        # Meta Q-table: meta_Q[phys_obs][subgoal_idx]
        self.meta_Q: Dict[tuple, Dict[int, float]] = {}
        # Controller Q-tables: ctrl_Q[sg_idx][phys_obs][action]
        self.ctrl_Q: Dict[int, Dict[tuple, Dict[str, float]]] = {
            i: {} for i in range(len(SUBGOALS))
        }

        # Option tracking
        self._sg: Optional[int] = None
        self._sg_start_obs: Optional[tuple] = None
        self._sg_steps: int = 0
        self._opt_ext_r: float = 0.0
        self._opt_discount: float = 1.0

        self.training_episodes: int = 0

    @staticmethod
    def get_physical_obs(env: ThreeCardMonteEnv) -> tuple:
        """Extract physical-only observation (5 features)."""
        s = env.state
        return (
            s.has("money_L"),
            s.has("money_C"),
            s.has("money_R"),
            s.has("face_up"),
            s.has("cards_moving"),
        )

    def begin_episode(self):
        """Reset option tracking at the start of an episode."""
        self._sg = None
        self._sg_start_obs = None
        self._sg_steps = 0
        self._opt_ext_r = 0.0
        self._opt_discount = 1.0

    def needs_option(self) -> bool:
        return self._sg is None

    def start_option(self, phys_obs: tuple) -> int:
        """Select subgoal via ε-greedy meta-controller."""
        if phys_obs not in self.meta_Q:
            self.meta_Q[phys_obs] = {i: self.q_init for i in range(len(SUBGOALS))}
        if random.random() < self.meta_epsilon:
            sg = random.randrange(len(SUBGOALS))
        else:
            sg = max(range(len(SUBGOALS)), key=lambda i: self.meta_Q[phys_obs].get(i, self.q_init))
        self._sg = sg
        self._sg_start_obs = phys_obs
        self._sg_steps = 0
        self._opt_ext_r = 0.0
        self._opt_discount = 1.0
        return sg

    def select_action(self, phys_obs: tuple, available: List[str]) -> str:
        """ε-greedy controller action for current subgoal."""
        sg = self._sg
        if phys_obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][phys_obs] = {a: self.q_init for a in self.all_actions}
        if random.random() < self.epsilon:
            return random.choice(available)
        qvals = self.ctrl_Q[sg][phys_obs]
        best_val = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == best_val]
        return random.choice(best)

    def select_action_greedy(self, phys_obs: tuple, sg: int, available: List[str]) -> str:
        """Greedy controller action (no exploration) for evaluation."""
        if phys_obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][phys_obs] = {a: self.q_init for a in self.all_actions}
        qvals = self.ctrl_Q[sg][phys_obs]
        best_val = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == best_val]
        return random.choice(best)

    def select_option_greedy(self, phys_obs: tuple) -> int:
        """Greedy meta-controller (no exploration) for evaluation."""
        if phys_obs not in self.meta_Q:
            self.meta_Q[phys_obs] = {i: self.q_init for i in range(len(SUBGOALS))}
        return max(range(len(SUBGOALS)), key=lambda i: self.meta_Q[phys_obs].get(i, self.q_init))

    def update(
        self,
        old_phys: tuple,
        action: str,
        new_phys: tuple,
        env: ThreeCardMonteEnv,
        extrinsic_reward: float,
        available_new: List[str],
        episode_done: bool,
    ) -> bool:
        """Update controller + meta-controller. Returns True if option terminated."""
        sg = self._sg
        self._sg_steps += 1

        # Check option termination
        achieved = SUBGOALS[sg]["check"](env)
        timeout = self._sg_steps >= SUBGOALS[sg]["max_steps"]
        option_done = achieved or timeout or episode_done

        # Controller Q-update with intrinsic reward
        r_int = self.intrinsic_bonus if achieved else -self.step_cost
        if old_phys not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][old_phys] = {a: self.q_init for a in self.all_actions}
        if new_phys not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][new_phys] = {a: self.q_init for a in self.all_actions}

        old_q = self.ctrl_Q[sg][old_phys].get(action, self.q_init)
        if option_done:
            td = r_int - old_q
        else:
            next_q = max(self.ctrl_Q[sg][new_phys].get(a, self.q_init) for a in available_new)
            td = r_int + self.gamma * next_q - old_q
        self.ctrl_Q[sg][old_phys][action] = old_q + self.lr * td

        # Accumulate discounted extrinsic reward
        self._opt_ext_r += self._opt_discount * extrinsic_reward
        self._opt_discount *= self.gamma

        # Meta SMDP update on option termination
        if option_done:
            start_obs = self._sg_start_obs
            if start_obs not in self.meta_Q:
                self.meta_Q[start_obs] = {i: self.q_init for i in range(len(SUBGOALS))}
            if new_phys not in self.meta_Q:
                self.meta_Q[new_phys] = {i: self.q_init for i in range(len(SUBGOALS))}

            k = self._sg_steps
            gamma_k = self.gamma ** k
            meta_target = self._opt_ext_r + gamma_k * max(self.meta_Q[new_phys].values())
            old_meta = self.meta_Q[start_obs].get(sg, self.q_init)
            self.meta_Q[start_obs][sg] = old_meta + self.lr * (meta_target - old_meta)

            self._sg = None

        return option_done
