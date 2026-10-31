"""
QRM + PBRS — Q-Learning for Reward Machines with Potential-Based Reward Shaping.

Augments the standard QRM agent with additional shaped rewards derived from a
potential function over RM states.  The shaping bonus on each transition is:

    F(s, s') = γ · Φ(u') − Φ(u)

where Φ(u) is the potential of RM state u.  By the Ng–Harada–Russell theorem
(ICML 1999), this form guarantees policy invariance: the optimal policy under
shaped rewards is identical to the optimal policy under the original rewards.

The practical benefit is faster convergence — the agent receives intermediate
credit for making progress through the RM phases even before reaching the
sparse +1 or +10 milestone rewards.

Potential function (RM-state based):
    Φ(U0) = 0.0   — start (no progress)
    Φ(U1) = 1.0   — observer saw ball under cup
    Φ(U2) = 2.0   — observer infers ball in hand
    Φ(U3) = 3.0   — ball appears / reveal
    Φ(UT) = 5.0   — deception milestone
    Φ(U4) = 4.0   — lime loaded (between U3 and UT)

References:
  Ng, A.Y., Harada, D. and Russell, S., 1999.
  "Policy Invariance Under Reward Transformations: Theory and Application to
  Reward Shaping." ICML 1999.

  Icarte, R.T. et al., 2018.
  "Using Reward Machines for High-Level Task Specification and Decomposition
  in Reinforcement Learning." ICML 2018.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from reward_machine import RewardMachine, RMState
from qrm import QRMAgent, State, EnvState, Action, _add_state_if_needed, _get_qmax


# ---------------------------------------------------------------------------
# Potential function over RM states
# ---------------------------------------------------------------------------

# Default potentials: higher = closer to milestone / more progress
RM_POTENTIALS: Dict[RMState, float] = {
    RMState.U0: 0.0,   # Setup
    RMState.U1: 1.0,   # Covered
    RMState.U2: 2.0,   # Pocket (observer inferred ball in hand)
    RMState.U3: 3.0,   # Appear (ball visible)
    RMState.UT: 5.0,   # Milestone — deception success
    RMState.U4: 4.0,   # Lime loaded (between U3 and next UT)
}


def phi(rm_state: RMState, potentials: Dict[RMState, float] = RM_POTENTIALS) -> float:
    """Return the potential of an RM state."""
    return potentials.get(rm_state, 0.0)


# ---------------------------------------------------------------------------
# QRM + PBRS Agent
# ---------------------------------------------------------------------------

class QRMPBRSAgent(QRMAgent):
    """QRM agent augmented with Potential-Based Reward Shaping.

    Inherits all QRM functionality.  The only change is in the update rule:
    the reward used for the Q-update is:

        r_shaped = r + γ · Φ(u') − Φ(u)

    where r is the original RM reward, Φ is the potential function over RM
    states, and γ is the discount factor.

    By the Ng–Harada–Russell theorem, this preserves the optimal policy
    while providing denser reward signal for faster convergence.
    """

    def __init__(
        self,
        reward_machine: RewardMachine,
        potentials: Optional[Dict[RMState, float]] = None,
        discount_factor: float = 0.9,
        learning_rate: float = 0.1,
        epsilon: float = 0.1,
        q_init: float = 0.0,
        all_actions: Optional[List[str]] = None,
    ):
        super().__init__(
            reward_machine=reward_machine,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            epsilon=epsilon,
            q_init=q_init,
            all_actions=all_actions,
        )
        self.potentials = potentials or RM_POTENTIALS

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
        """Q-update with potential-based shaping bonus.

        r_shaped = r + γ · Φ(u') − Φ(u)

        Then standard QRM update:
            Q[obs][a] += α · (r_shaped + γ · max_a' Q[obs'][a'] − Q[obs][a])
        """
        # Compute shaping bonus
        shaping_bonus = (
            self.gamma * phi(next_rm_state, self.potentials)
            - phi(rm_state, self.potentials)
        )
        shaped_reward = reward + shaping_bonus

        # Delegate to parent update with shaped reward
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)

        obs_next = self._make_obs(next_env_state, next_rm_state)
        _add_state_if_needed(self.Q, obs_next, self.all_actions, self.q_init)
        qmax_next = _get_qmax(self.Q, obs_next, next_available_actions, self.q_init)
        delta = shaped_reward + self.gamma * qmax_next - self.Q[obs].get(action, self.q_init)

        self.Q[obs][action] = self.Q[obs].get(action, self.q_init) + self.lr * delta

        self._step_count += 1
        self._transition_count += 1
