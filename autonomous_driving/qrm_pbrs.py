"""BEQRM+PBRS — QRM with Potential-Based Reward Shaping (Autonomous Driving).

Potential function Φ(s, u) combines RM-state shaping with positional shaping:
    Φ(s, u) = Φ_RM(u) + Φ_pos(x, y)

RM potentials over the 4-state Waymo RM (u0-u3, uT, uF):
    u0: start         Φ_RM=0.0
    u1: navigating    Φ_RM=0.3
    u2: ambiguous     Φ_RM=0.5
    u3: resolved      Φ_RM=0.7
    uT: success       Φ_RM=0.0
    uF: failure       Φ_RM=0.0

Positional potential decreases with Manhattan distance to goal:
    Φ_pos(x, y) = α · (1 - manhattan((x,y), GOAL) / max_dist)

Adding a bounded positional potential preserves optimality (Ng, Harada,
Russell 1999) — the telescoping argument holds for any bounded Φ.

Shaped reward: R'(s, a, s') = R(s, a, s') + γ·Φ(s', u') - Φ(s, u)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from reward_machine import RewardMachine, RMState
from qrm import QRMAgent, _add_state_if_needed, _get_qmax
from autonomous_driving_env import GOAL, GRID_SIZE

# RM-state potentials (Waymo 4-state RM: U0-U3, UT, UF)
RM_POTENTIALS: Dict[RMState, float] = {
    RMState.U0: 0.0,
    RMState.U1: 0.3,
    RMState.U2: 0.5,
    RMState.U3: 0.7,
    RMState.UT: 0.0,
    RMState.UF: 0.0,
}

# Positional shaping scale (bounded — preserves optimality). Tuned down to
# 1.0 for the smaller 4x4 grid so positional gradient does not dominate the
# more nuanced rule-compliance subgoal.
POS_POTENTIAL_SCALE = 1.0
MAX_DIST = 2 * (GRID_SIZE - 1)  # worst-case Manhattan distance


def _pos_potential(phys_obs: tuple) -> float:
    """Positional shaping based on Manhattan distance to goal.

    phys_obs = (x, y, heading_idx, visual_ahead_idx,
                traffic_pass, traffic_avoid, traffic_slow,
                safe, legal, reached_B).
    Only the first two elements (x, y) are used.
    """
    x = phys_obs[0]
    y = phys_obs[1]
    gx, gy = GOAL
    dist = abs(x - gx) + abs(y - gy)
    return POS_POTENTIAL_SCALE * (1.0 - dist / MAX_DIST)


class QRMPBRSAgent(QRMAgent):
    """BEQRM + Potential-Based Reward Shaping (RM + positional)."""

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

    def _potential(self, env_state: tuple, rm_state: RMState) -> float:
        # env_state = (phys_tuple, beliefs_tuple); phys is index 0
        phys = env_state[0]
        return self.potentials.get(rm_state, 0.0) + _pos_potential(phys)

    def update(self, env_state, action, next_env_state, rm_state, next_rm_state, reward, next_available_actions):
        """Counterfactual QRM update with PBRS-shaped reward at each RM state.

        For each possible RM state u, query u → u' via the RM and apply the
        Q-update with shaped reward R + γ·Φ(s',u') − Φ(s,u). Mirrors the
        counterfactual update in QRMAgent but with potential-based shaping.
        """
        post_env_state = getattr(self, "_post_true_props", None)
        if post_env_state is None:
            # Fallback: legacy single-state shaped update
            phi_old = self._potential(env_state, rm_state)
            phi_new = self._potential(next_env_state, next_rm_state)
            shaped_reward = reward + self.gamma * phi_new - phi_old
            obs = self._make_obs(env_state, rm_state)
            _add_state_if_needed(self.Q, obs, self.all_actions, self.q_init)
            obs_next = self._make_obs(next_env_state, next_rm_state)
            _add_state_if_needed(self.Q, obs_next, self.all_actions, self.q_init)
            qmax_next = _get_qmax(self.Q, obs_next, next_available_actions, self.q_init)
            old_q = self.Q[obs].get(action, self.q_init)
            self.Q[obs][action] = old_q + self.lr * (
                shaped_reward + self.gamma * qmax_next - old_q
            )
            self._step_count += 1
            return

        for u in self.rm.get_all_states():
            u_next, r_u = self.rm.query_transition(u, post_env_state)
            phi_old = self._potential(env_state, u)
            phi_new = self._potential(next_env_state, u_next)
            shaped_r = r_u + self.gamma * phi_new - phi_old
            obs_u = self._make_obs(env_state, u)
            _add_state_if_needed(self.Q, obs_u, self.all_actions, self.q_init)
            old_q = self.Q[obs_u].get(action, self.q_init)
            if u_next == RMState.UT:
                # Terminal: no bootstrap
                target = shaped_r
            else:
                obs_u_next = self._make_obs(next_env_state, u_next)
                _add_state_if_needed(self.Q, obs_u_next, self.all_actions, self.q_init)
                qmax_next = _get_qmax(self.Q, obs_u_next, next_available_actions, self.q_init)
                target = shaped_r + self.gamma * qmax_next
            self.Q[obs_u][action] = old_q + self.lr * (target - old_q)
        self._step_count += 1
