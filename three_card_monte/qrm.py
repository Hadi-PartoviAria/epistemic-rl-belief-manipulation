"""BEQRM — Belief-Enhanced Q-Learning for Reward Machines (Three-Card Monte).

Maintains Q-tables indexed by (env_state, rm_state).
Only the current RM state's Q-values are updated per step.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from reward_machine import RewardMachine, RMState

State = tuple
EnvState = tuple
Action = str


def _get_qmax(Q: dict, s: State, actions: List[Action], q_init: float) -> float:
    if s not in Q:
        Q[s] = {a: q_init for a in actions}
    return max(Q[s].values()) if Q[s] else 0.0


def _get_best_action(Q: dict, s: State, actions: List[Action], q_init: float) -> Action:
    if s not in Q:
        Q[s] = {a: q_init for a in actions}
    qmax = max(Q[s].get(a, q_init) for a in actions)
    best = [a for a in actions if Q[s].get(a, q_init) == qmax]
    return random.choice(best) if best else random.choice(actions)


def _add_state_if_needed(Q: dict, s: State, actions: List[Action], q_init: float) -> None:
    if s not in Q:
        Q[s] = {a: q_init for a in actions}


class QRMAgent:
    """BEQRM agent for Three-Card Monte."""

    def __init__(
        self,
        reward_machine: RewardMachine,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        q_init: float = 0.0,
        all_actions: Optional[List[str]] = None,
    ):
        self.rm = reward_machine
        self.gamma = discount_factor
        self.lr = learning_rate
        self.epsilon = epsilon
        self.q_init = q_init

        from three_card_env import ALL_ACTIONS
        self.all_actions = all_actions or ALL_ACTIONS

        self.Q: Dict[State, Dict[Action, float]] = {}
        self.training_episodes: int = 0
        self._step_count: int = 0

    def _make_obs(self, env_state: EnvState, rm_state: RMState) -> State:
        return (env_state, rm_state.value)

    def select_action(self, env_state: EnvState, rm_state: RMState, available_actions: List[Action]) -> Action:
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        return _get_best_action(self.Q, obs, available_actions, self.q_init)

    def get_policy_action(self, env_state: EnvState, rm_state: RMState, available_actions: List[Action]) -> Action:
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)
        return _get_best_action(self.Q, obs, available_actions, self.q_init)

    def update(self, env_state, action, next_env_state, rm_state, next_rm_state, reward, next_available_actions):
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)
        obs_next = self._make_obs(next_env_state, next_rm_state)
        _add_state_if_needed(self.Q, obs_next, self.all_actions, self.q_init)
        qmax_next = _get_qmax(self.Q, obs_next, next_available_actions, self.q_init)
        delta = reward + self.gamma * qmax_next - self.Q[obs].get(action, self.q_init)
        self.Q[obs][action] = self.Q[obs].get(action, self.q_init) + self.lr * delta
        self._step_count += 1

    @property
    def num_states(self) -> int:
        return len(self.Q)
