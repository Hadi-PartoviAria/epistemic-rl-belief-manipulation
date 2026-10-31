"""DDQN — Double Deep Q-Network Baseline for Three-Card Monte.

Neural network baseline using a two-layer MLP to approximate Q(s,a).
Uses the same epistemic observation as BEQRM (physical + beliefs + RM index).
State vector: 12 floats (5 physical + 6 beliefs + 1 RM index).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from reward_machine import RewardMachine, RMState
from three_card_env import ThreeCardMonteEnv, ALL_ACTIONS


class QNetwork(nn.Module):
    """Two-layer MLP for Q-value approximation."""

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


class Transition(NamedTuple):
    state: np.ndarray
    action_idx: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def _obs_to_vector(env: ThreeCardMonteEnv) -> np.ndarray:
    """Convert environment state to a flat float vector (12 dims).

    [0..4]  physical: money_L, money_C, money_R, face_up, cards_moving
    [5..10] beliefs: eb_money_L, eb_not_money_L, ..., eb_money_R, eb_not_money_R
    [11]    RM state index / 3.0 (normalised)
    """
    s = env.state
    phys = [
        float(s.has("money_L")),
        float(s.has("money_C")),
        float(s.has("money_R")),
        float(s.has("face_up")),
        float(s.has("cards_moving")),
    ]
    beliefs = [
        float(s.has("eb_money_L")),
        float(s.has("eb_not_money_L")),
        float(s.has("eb_money_C")),
        float(s.has("eb_not_money_C")),
        float(s.has("eb_money_R")),
        float(s.has("eb_not_money_R")),
    ]
    rm_idx = float(env.reward_machine.get_state_index()) / 3.0  # 4 RM states

    return np.array(phys + beliefs + [rm_idx], dtype=np.float32)


class DDQNAgent:
    """Double Deep Q-Network agent for Three-Card Monte."""

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
        state_dim: int = 12,
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
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self._step_count / max(self.epsilon_decay, 1))

    def select_action(self, state_vec: np.ndarray, available: List[str]) -> str:
        if random.random() < self.epsilon:
            return random.choice(available)
        return self._greedy_action(state_vec, available)

    def get_policy_action(self, state_vec: np.ndarray, available: List[str]) -> str:
        return self._greedy_action(state_vec, available)

    def _greedy_action(self, state_vec: np.ndarray, available: List[str]) -> str:
        with torch.no_grad():
            q_vals = self.online_net(
                torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0).cpu().numpy()
        avail_indices = [self.action_to_idx[a] for a in available]
        best_idx = max(avail_indices, key=lambda i: q_vals[i])
        return self.idx_to_action[best_idx]

    def store_transition(self, state_vec, action, reward, next_state_vec, done):
        action_idx = self.action_to_idx[action]
        self.buffer.push(state_vec, action_idx, reward, next_state_vec, done)
        self._step_count += 1

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action_idx for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        # Online net selects actions
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Sync target network
        if self._step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
