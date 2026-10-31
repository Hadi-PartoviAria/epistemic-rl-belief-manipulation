"""
QRM — Q-Learning for Reward Machines (Icarte et al., ICML 2018).

Follows the reference implementation:
  https://github.com/RodrigoToroIcarte/reward_machines

Core algorithm (CRM — Counterfactual experiences for Reward Machines):
  For each real transition (s, a, s') observed in the environment:
    For EVERY RM state u ∈ U:
      1. Query RM:  (u', r_u) = δ(u, label(s'))
      2. Construct counterfactual experience:
         obs_u  = (s, u)          [env state with RM state u]
         obs_u' = (s', u')        [next env state with next RM state u']
      3. Online Q-update (no terminal — episodes end by step limit):
         Q[obs_u][a] += α · (r_u + γ · max_a' Q[obs_u'][a'] - Q[obs_u][a])

This produces |U| Q-table updates per real transition, accelerating learning.

Key differences from batch model-based Q-learning:
  - Online incremental updates (learning rate α), NOT batch full-sweep
  - Standard ε-greedy exploration, NO prefer-unseen heuristic
  - Optimistic Q-initialization (q_init > 0) to encourage exploration
  - Single Q-table over cross-product (env_state × rm_state)

Reference:
  Icarte, R.T., Klassen, T.Q., Valenzano, R. and McIlraith, S.A., 2018.
  "Using Reward Machines for High-Level Task Specification and Decomposition
  in Reinforcement Learning." ICML 2018.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from reward_machine import RewardMachine, RMState


# Type aliases
State = tuple       # hashable observation (env_state, rm_state_index)
EnvState = tuple    # hashable env observation (without RM state)
Action = str


def _get_qmax(Q: dict, s: State, actions: List[Action], q_init: float) -> float:
    """Return max Q-value over available actions (matching reference impl)."""
    if s not in Q:
        Q[s] = {a: q_init for a in actions}
    return max(Q[s].values()) if Q[s] else 0.0


def _get_best_action(Q: dict, s: State, actions: List[Action], q_init: float) -> Action:
    """Select the action with highest Q-value, breaking ties randomly."""
    if s not in Q:
        Q[s] = {a: q_init for a in actions}
    qmax = max(Q[s].get(a, q_init) for a in actions)
    best = [a for a in actions if Q[s].get(a, q_init) == qmax]
    return random.choice(best) if best else random.choice(actions)


def _add_state_if_needed(Q: dict, s: State, actions: List[Action], q_init: float) -> None:
    """Initialize Q-values for a new state if not yet seen."""
    if s not in Q:
        Q[s] = {a: q_init for a in actions}


class QRMAgent:
    """Q-Learning for Reward Machines (QRM).

    Maintains separate Q-functions Q_u(s, a) per RM state u.
    Implemented as a single Q-table indexed by (env_state, rm_state).
    Only the current RM state's Q-values are updated per step.
    Online incremental updates with learning rate α.
    """

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

        # All possible actions (for Q-table initialization)
        self.all_actions = all_actions or [
            "lift_cup", "place_cup", "palm_ball", "reveal_hand", "load_lime"
        ]

        # Single Q-table: Q[(env_state, rm_index)][action] = q_value
        self.Q: Dict[State, Dict[Action, float]] = {}

        self.training_episodes: int = 0
        self._step_count: int = 0
        self._transition_count: int = 0

    def _make_obs(self, env_state: EnvState, rm_state: RMState) -> State:
        """Construct cross-product observation (env_state, rm_index)."""
        return (env_state, rm_state.value)

    def select_action(
        self,
        env_state: EnvState,
        rm_state: RMState,
        available_actions: List[Action],
    ) -> Action:
        """ε-greedy action selection (matching reference implementation).

        With probability ε: uniform random action.
        Otherwise: greedy (best Q-value with random tie-breaking).
        """
        if not available_actions:
            raise ValueError("No available actions")

        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)

        if random.random() < self.epsilon:
            return random.choice(available_actions)
        return _get_best_action(self.Q, obs, available_actions, self.q_init)

    def get_policy_action(
        self,
        env_state: EnvState,
        rm_state: RMState,
        available_actions: List[Action],
    ) -> Action:
        """Pure greedy action (no exploration) for evaluation."""
        if not available_actions:
            raise ValueError("No available actions")
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)
        return _get_best_action(self.Q, obs, available_actions, self.q_init)

    def update(
        self,
        env_state: EnvState,
        action: Action,
        next_env_state: EnvState,
        rm_state: RMState,
        next_rm_state: RMState,
        reward: float,
        next_available_actions: List[Action],
    ) -> None:
        """Online Q-update for the current RM state only (pure QRM).

        Updates Q_u(s, a) for the current RM state u:
          1. obs   = (env_state, u)
             obs'  = (next_env_state, u')
          2. No terminal RM state — episodes end by step limit only:
                δ = r + γ · max_a' Q[obs'][a'] - Q[obs][a]
          3. Q[obs][a] += α · δ
        """
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)

        obs_next = self._make_obs(next_env_state, next_rm_state)
        _add_state_if_needed(self.Q, obs_next, self.all_actions, self.q_init)
        qmax_next = _get_qmax(self.Q, obs_next, next_available_actions, self.q_init)
        delta = reward + self.gamma * qmax_next - self.Q[obs].get(action, self.q_init)

        self.Q[obs][action] = self.Q[obs].get(action, self.q_init) + self.lr * delta

        self._step_count += 1
        self._transition_count += 1

    # -------------------------------------------------------------------
    # Properties & diagnostics
    # -------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        """Number of unique states in the Q-table."""
        return len(self.Q)

    @property
    def num_transitions(self) -> int:
        return self._transition_count

    def get_q_table_summary(self) -> str:
        """Human-readable Q-table summary."""
        lines = []
        for state in sorted(self.Q.keys(), key=str):
            for action, q_val in sorted(self.Q[state].items()):
                if abs(q_val - self.q_init) > 1e-6:  # Only show non-default
                    lines.append(f"    Q({state}, {action}) = {q_val:.4f}")
        return "\n".join(lines[:100]) if lines else "    (empty — all at q_init)"
