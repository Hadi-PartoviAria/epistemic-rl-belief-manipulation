"""
DDQN — Double Deep Q-Network baseline for the Chop Cup.

Uses a neural network to approximate Q(s, a) instead of a tabular Q-table.
The observation is the same epistemic observation as QRM (physical state +
explicit beliefs + RM state index), but encoded as a float vector.

Double DQN (van Hasselt et al., 2016):
  - Online network selects action:   a* = argmax_a Q_online(s', a)
  - Target network evaluates value:  y  = r + γ · Q_target(s', a*)
  - This decouples selection from evaluation, reducing overestimation bias.

Reference:
  van Hasselt, H., Guez, A. and Silver, D., 2016.
  "Deep Reinforcement Learning with Double Q-Learning." AAAI 2016.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_machine import RewardMachine, RMState
from chop_cup_env import ChopCupEnv


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Simple MLP for Q-value approximation."""

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DDQN Agent
# ---------------------------------------------------------------------------

ALL_ACTIONS = ["lift_cup", "place_cup", "palm_ball", "reveal_hand", "load_lime"]


def _obs_to_vector(env: ChopCupEnv) -> np.ndarray:
    """Convert environment state to a float vector for the neural network.

    Layout (19 features):
      [0..9]   physical fluents (10 bools + lime_count normalised)
      [10..17] explicit belief fluents (8 bools)
      [18]     RM state index (normalised to [0, 1])
    """
    if env.state is None:
        return np.zeros(19, dtype=np.float32)

    s = env.state
    phys = [
        float(s.has("mag_stuck")),
        float(s.has("mag_visible")),
        float(s.has("reg_under")),
        float(s.has("reg_table")),
        float(s.has("reg_hand")),
        float(s.has("cup_down")),
        float(s.has("ball_visible")),
        float(s.has("lime_cup")),
        float(s.has("lime_visible")),
        float(env.lime_count) / 5.0,  # normalise
    ]

    beliefs = [
        float(s.has("eb_reg_under")),
        float(s.has("eb_not_reg_under")),
        float(s.has("eb_reg_table")),
        float(s.has("eb_not_reg_table")),
        float(s.has("eb_reg_hand")),
        float(s.has("eb_not_reg_hand")),
        float(s.has("eb_lime_cup")),
        float(s.has("eb_not_lime_cup")),
    ]

    rm_idx = float(env.reward_machine.get_state_index()) / 5.0  # 6 RM states

    return np.array(phys + beliefs + [rm_idx], dtype=np.float32)


class DDQNAgent:
    """Double Deep Q-Network agent for the Chop Cup.

    Uses the same epistemic observation space as QRM but approximates
    Q-values with a neural network instead of a lookup table.
    """

    def __init__(
        self,
        reward_machine: RewardMachine,
        discount: float = 0.9,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 500,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update: int = 100,
        state_dim: int = 19,
        hidden: int = 128,
    ):
        self.rm = reward_machine
        self.gamma = discount
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.num_actions = len(ALL_ACTIONS)
        self.action_to_idx = {a: i for i, a in enumerate(ALL_ACTIONS)}
        self.idx_to_action = {i: a for i, a in enumerate(ALL_ACTIONS)}

        self.device = torch.device("cpu")

        # Networks
        self.online_net = QNetwork(state_dim, self.num_actions, hidden).to(self.device)
        self.target_net = QNetwork(state_dim, self.num_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.training_episodes: int = 0
        self._step_count: int = 0

    @property
    def epsilon(self) -> float:
        """Current exploration rate with exponential decay."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self._step_count / max(self.epsilon_decay, 1))

    def select_action(self, state_vec: np.ndarray, available: List[str]) -> str:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(available)
        return self._greedy_action(state_vec, available)

    def get_policy_action(self, state_vec: np.ndarray, available: List[str]) -> str:
        """Pure greedy action (no exploration) for evaluation."""
        return self._greedy_action(state_vec, available)

    def _greedy_action(self, state_vec: np.ndarray, available: List[str]) -> str:
        """Select best action from available set."""
        with torch.no_grad():
            q_vals = self.online_net(
                torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0).cpu().numpy()

        # Mask unavailable actions
        avail_indices = [self.action_to_idx[a] for a in available]
        best_idx = max(avail_indices, key=lambda i: q_vals[i])
        return self.idx_to_action[best_idx]

    def store_transition(
        self,
        state_vec: np.ndarray,
        action: str,
        reward: float,
        next_state_vec: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        action_idx = self.action_to_idx[action]
        self.buffer.push(state_vec, action_idx, reward, next_state_vec, done)
        self._step_count += 1

    def train_step(self) -> Optional[float]:
        """Sample a batch and perform one gradient step (Double DQN).

        Returns the loss value, or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)

        # Current Q-values
        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            online_next_q = self.online_net(next_states_t)
            best_actions = online_next_q.argmax(dim=1, keepdim=True)
            target_next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            targets = rewards_t + self.gamma * target_next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically update target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
