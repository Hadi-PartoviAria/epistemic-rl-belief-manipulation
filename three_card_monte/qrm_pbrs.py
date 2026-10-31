"""BEQRM+PBRS — QRM with Potential-Based Reward Shaping (Three-Card Monte).

Potential function Φ(u) over RM states, designed for the cyclic RM
(u0 → u1 → u2 → u3 → u0):

    Φ(u0) = 0,  Φ(u1) = 0.5,  Φ(u2) = 0.8,  Φ(u3) = 0

The potential increases along the forward path (u0→u1→u2) to provide
intermediate credit for phase transitions, then drops to zero at u3
so the loop transition u3→u0 incurs no penalty.  Values are kept
moderate so the net shaped reward at the key transition c (u2→u3)
remains positive: R' = 1.0 + γ·0 − 0.8 = +0.28.

Shaped reward: R'(s,a,s') = R(s,a,s') + γ·Φ(u') - Φ(u)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from reward_machine import RewardMachine, RMState
from qrm import QRMAgent

# Cycle-aware potentials: ramp up along forward path, reset at u3
RM_POTENTIALS: Dict[RMState, float] = {
    RMState.U0: 0.0,
    RMState.U1: 0.5,
    RMState.U2: 0.8,
    RMState.U3: 0.0,
}


class QRMPBRSAgent(QRMAgent):
    """BEQRM + Potential-Based Reward Shaping."""

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

    def update(self, env_state, action, next_env_state, rm_state, next_rm_state, reward, next_available_actions):
        """Q-update with shaped reward: R + γ·Φ(u') - Φ(u)."""
        phi_old = self.potentials.get(rm_state, 0.0)
        phi_new = self.potentials.get(next_rm_state, 0.0)
        shaping_bonus = self.gamma * phi_new - phi_old
        shaped_reward = reward + shaping_bonus

        # Standard Q-update with shaped reward
        from qrm import _add_state_if_needed, _get_qmax
        obs = self._make_obs(env_state, rm_state)
        _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)
        obs_next = self._make_obs(next_env_state, next_rm_state)
        _add_state_if_needed(self.Q, obs_next, self.all_actions, self.q_init)
        qmax_next = _get_qmax(self.Q, obs_next, next_available_actions, self.q_init)
        delta = shaped_reward + self.gamma * qmax_next - self.Q[obs].get(action, self.q_init)
        self.Q[obs][action] = self.Q[obs].get(action, self.q_init) + self.lr * delta
        self._step_count += 1
