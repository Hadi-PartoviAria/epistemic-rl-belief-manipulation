"""
HERL-QRM — Hierarchical Epistemic RL with Reward Machines.

Combines the best of both worlds:
  - Hierarchical options (macro-actions) from HRL
  - Epistemic reasoning (explicit/implicit beliefs) from the BB-POMDP framework
  - Reward Machine structure from QRM (Icarte et al., 2018)

Two-level hierarchy mirroring HRLAgent but with epistemic observations:
  Meta-controller: SMDP Q-learning over subgoals, indexed by
                   (epistemic_obs, rm_state, subgoal)
  Controller:      Q-learning for primitive actions per subgoal, indexed by
                   (epistemic_obs, rm_state, action)

Key difference from plain QRM:
  - Actions are temporally extended options, not primitive actions
  - SMDP Q-learning with γ^k discounting for multi-step options

Key difference from plain HRL:
  - Full epistemic observation (physical + beliefs + RM state)
  - RM-structured Q-tables: Q_u(s, o) per RM state u
  - SAT-based implicit belief reasoning informs option selection

This represents the "full integration" approach: hierarchical temporal
abstraction + epistemic reasoning + reward machine structure.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from chop_cup_env import ChopCupEnv
from reward_machine import RewardMachine, RMState
from hrl_agent import SUBGOALS


# ---------------------------------------------------------------------------
# HRL-QRM Agent
# ---------------------------------------------------------------------------

class HRLQRMAgent:
    """Hierarchical Epistemic RL with Reward Machine.

    Like HRLAgent, but observations include belief fluents AND RM state.
    Uses the same subgoals as HRL for fair comparison.
    """

    ALL_ACTIONS = ["lift_cup", "place_cup", "palm_ball", "reveal_hand", "load_lime"]

    def __init__(
        self,
        reward_machine: RewardMachine,
        discount: float = 0.9,
        lr: float = 0.1,
        epsilon: float = 0.1,
        meta_epsilon: float = 0.2,
        q_init: float = 0.0,
        intrinsic_bonus: float = 1.0,
        step_cost: float = 0.01,
    ):
        self.rm = reward_machine
        self.gamma = discount
        self.lr = lr
        self.epsilon = epsilon
        self.meta_epsilon = meta_epsilon
        self.q_init = q_init
        self.intrinsic_bonus = intrinsic_bonus
        self.step_cost = step_cost
        self.num_subgoals = len(SUBGOALS)

        # Meta Q-table: meta_Q[(epistemic_obs, rm_state)][subgoal_idx]
        self.meta_Q: Dict[tuple, Dict[int, float]] = {}

        # Controller Q-tables (one per subgoal):
        # ctrl_Q[sg][(epistemic_obs, rm_state)][action]
        self.ctrl_Q: List[Dict[tuple, Dict[str, float]]] = [
            {} for _ in range(self.num_subgoals)
        ]

        # Option tracking
        self._sg: Optional[int] = None
        self._opt_steps: int = 0
        self._opt_start: Optional[tuple] = None
        self._opt_ext_r: float = 0.0
        self._opt_gamma: float = 1.0

        self.training_episodes: int = 0
        self._transition_count: int = 0

    def _make_obs(self, env: ChopCupEnv) -> tuple:
        """Full epistemic observation + RM state."""
        return (env.get_env_observation(), env.reward_machine.get_state().value)

    def _init_meta(self, obs: tuple) -> None:
        if obs not in self.meta_Q:
            self.meta_Q[obs] = {i: self.q_init for i in range(self.num_subgoals)}

    def _init_ctrl(self, sg: int, obs: tuple) -> None:
        if obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][obs] = {a: self.q_init for a in self.ALL_ACTIONS}

    # --- Training lifecycle ---

    def begin_episode(self) -> None:
        self._sg = None
        self._opt_steps = 0
        self._opt_start = None
        self._opt_ext_r = 0.0
        self._opt_gamma = 1.0

    def needs_option(self) -> bool:
        return self._sg is None

    def start_option(self, env: ChopCupEnv) -> int:
        obs = self._make_obs(env)
        self._init_meta(obs)

        if random.random() < self.meta_epsilon:
            sg = random.randint(0, self.num_subgoals - 1)
        else:
            qvals = self.meta_Q[obs]
            maxq = max(qvals.values())
            best = [i for i, q in qvals.items() if q == maxq]
            sg = random.choice(best)

        self._sg = sg
        self._opt_steps = 0
        self._opt_start = obs
        self._opt_ext_r = 0.0
        self._opt_gamma = 1.0
        return sg

    def select_action(self, env: ChopCupEnv, available: List[str]) -> str:
        obs = self._make_obs(env)
        sg = self._sg
        self._init_ctrl(sg, obs)

        if random.random() < self.epsilon:
            return random.choice(available)

        qvals = self.ctrl_Q[sg][obs]
        maxq = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == maxq]
        return random.choice(best)

    def update(
        self,
        old_env: ChopCupEnv,
        old_obs: tuple,
        action: str,
        new_obs: tuple,
        env_state,
        extrinsic_reward: float,
        available_new: List[str],
        episode_done: bool,
    ) -> bool:
        """Process one step. Returns True if option terminated."""
        sg = self._sg
        self._opt_steps += 1
        self._transition_count += 1

        achieved = SUBGOALS[sg].check(env_state)
        timeout = self._opt_steps >= SUBGOALS[sg].max_steps
        option_done = achieved or timeout or episode_done

        # Controller update (intrinsic reward)
        self._init_ctrl(sg, old_obs)
        self._init_ctrl(sg, new_obs)

        r_int = self.intrinsic_bonus if achieved else -self.step_cost
        old_q = self.ctrl_Q[sg][old_obs].get(action, self.q_init)

        if option_done:
            td = r_int - old_q
        else:
            acts = available_new if available_new else self.ALL_ACTIONS
            maxq_next = max(self.ctrl_Q[sg][new_obs].get(a, self.q_init) for a in acts)
            td = r_int + self.gamma * maxq_next - old_q

        self.ctrl_Q[sg][old_obs][action] = old_q + self.lr * td

        # Accumulate extrinsic reward
        self._opt_ext_r += self._opt_gamma * extrinsic_reward
        self._opt_gamma *= self.gamma

        # Meta-controller SMDP update on option termination
        if option_done:
            start = self._opt_start
            self._init_meta(start)
            self._init_meta(new_obs)

            k = self._opt_steps
            R = self._opt_ext_r

            old_meta_q = self.meta_Q[start][sg]
            if episode_done:
                meta_target = R
            else:
                meta_target = R + (self.gamma ** k) * max(self.meta_Q[new_obs].values())

            self.meta_Q[start][sg] = old_meta_q + self.lr * (meta_target - old_meta_q)
            self._sg = None

        return option_done

    # --- Evaluation (greedy) ---

    def select_option_greedy(self, env: ChopCupEnv) -> int:
        obs = self._make_obs(env)
        self._init_meta(obs)
        maxq = max(self.meta_Q[obs].values())
        best = [i for i, q in self.meta_Q[obs].items() if q == maxq]
        return random.choice(best)

    def select_action_greedy(self, env: ChopCupEnv, sg: int, available: List[str]) -> str:
        obs = self._make_obs(env)
        self._init_ctrl(sg, obs)
        qvals = self.ctrl_Q[sg][obs]
        maxq = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == maxq]
        return random.choice(best)

    @property
    def num_states(self) -> int:
        total = len(self.meta_Q)
        for cq in self.ctrl_Q:
            total += len(cq)
        return total

    @property
    def num_transitions(self) -> int:
        return self._transition_count
