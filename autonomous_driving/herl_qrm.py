"""HRL-RM — Hierarchical RL with Reward Machine + Epistemic Observations
   (Autonomous Driving).

Same option hierarchy as HRL, but observations include epistemic beliefs + RM state.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from autonomous_driving_env import AutonomousDrivingEnv, ALL_ACTIONS
from reward_machine import RewardMachine, RMState
from hrl_agent import SUBGOALS


class HRLQRMAgent:
    """HRL-RM agent with epistemic + RM observations."""

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
        self.all_actions = ALL_ACTIONS

        self.meta_Q: Dict[tuple, Dict[int, float]] = {}
        self.ctrl_Q: Dict[int, Dict[tuple, Dict[str, float]]] = {
            i: {} for i in range(len(SUBGOALS))
        }

        self._sg: Optional[int] = None
        self._sg_start_obs: Optional[tuple] = None
        self._sg_steps: int = 0
        self._opt_ext_r: float = 0.0
        self._opt_discount: float = 1.0

        self.training_episodes: int = 0

    def _make_obs(self, env: AutonomousDrivingEnv) -> tuple:
        """Full epistemic observation + RM state."""
        return (env.get_env_observation(), env.reward_machine.get_state().value)

    def begin_episode(self):
        self._sg = None
        self._sg_start_obs = None
        self._sg_steps = 0
        self._opt_ext_r = 0.0
        self._opt_discount = 1.0

    def needs_option(self) -> bool:
        return self._sg is None

    def start_option(self, env: AutonomousDrivingEnv) -> int:
        obs = self._make_obs(env)
        if obs not in self.meta_Q:
            self.meta_Q[obs] = {i: self.q_init for i in range(len(SUBGOALS))}
        if random.random() < self.meta_epsilon:
            sg = random.randrange(len(SUBGOALS))
        else:
            sg = max(range(len(SUBGOALS)), key=lambda i: self.meta_Q[obs].get(i, self.q_init))
        self._sg = sg
        self._sg_start_obs = obs
        self._sg_steps = 0
        self._opt_ext_r = 0.0
        self._opt_discount = 1.0
        return sg

    def select_action(self, env: AutonomousDrivingEnv, available: List[str]) -> str:
        sg = self._sg
        obs = self._make_obs(env)
        if obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][obs] = {a: self.q_init for a in self.all_actions}
        if random.random() < self.epsilon:
            return random.choice(available)
        qvals = self.ctrl_Q[sg][obs]
        best_val = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == best_val]
        return random.choice(best)

    def select_action_greedy(self, env: AutonomousDrivingEnv, sg: int, available: List[str]) -> str:
        obs = self._make_obs(env)
        if obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][obs] = {a: self.q_init for a in self.all_actions}
        qvals = self.ctrl_Q[sg][obs]
        best_val = max(qvals.get(a, self.q_init) for a in available)
        best = [a for a in available if qvals.get(a, self.q_init) == best_val]
        return random.choice(best)

    def select_option_greedy(self, env: AutonomousDrivingEnv) -> int:
        obs = self._make_obs(env)
        if obs not in self.meta_Q:
            self.meta_Q[obs] = {i: self.q_init for i in range(len(SUBGOALS))}
        return max(range(len(SUBGOALS)), key=lambda i: self.meta_Q[obs].get(i, self.q_init))

    def update(
        self,
        old_env: AutonomousDrivingEnv,
        old_obs: tuple,
        action: str,
        new_obs: tuple,
        env: AutonomousDrivingEnv,
        extrinsic_reward: float,
        available_new: List[str],
        episode_done: bool,
    ) -> bool:
        sg = self._sg
        self._sg_steps += 1

        achieved = SUBGOALS[sg]["check"](env)
        timeout = self._sg_steps >= SUBGOALS[sg]["max_steps"]
        option_done = achieved or timeout or episode_done

        r_int = self.intrinsic_bonus if achieved else -self.step_cost
        if old_obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][old_obs] = {a: self.q_init for a in self.all_actions}
        if new_obs not in self.ctrl_Q[sg]:
            self.ctrl_Q[sg][new_obs] = {a: self.q_init for a in self.all_actions}

        old_q = self.ctrl_Q[sg][old_obs].get(action, self.q_init)
        if option_done:
            td = r_int - old_q
        else:
            next_q = max(self.ctrl_Q[sg][new_obs].get(a, self.q_init) for a in available_new)
            td = r_int + self.gamma * next_q - old_q
        self.ctrl_Q[sg][old_obs][action] = old_q + self.lr * td

        self._opt_ext_r += self._opt_discount * extrinsic_reward
        self._opt_discount *= self.gamma

        if option_done:
            start_obs = self._sg_start_obs
            if start_obs not in self.meta_Q:
                self.meta_Q[start_obs] = {i: self.q_init for i in range(len(SUBGOALS))}
            if new_obs not in self.meta_Q:
                self.meta_Q[new_obs] = {i: self.q_init for i in range(len(SUBGOALS))}
            k = self._sg_steps
            gamma_k = self.gamma ** k
            meta_target = self._opt_ext_r + gamma_k * max(self.meta_Q[new_obs].values())
            old_meta = self.meta_Q[start_obs].get(sg, self.q_init)
            self.meta_Q[start_obs][sg] = old_meta + self.lr * (meta_target - old_meta)
            self._sg = None

        return option_done
