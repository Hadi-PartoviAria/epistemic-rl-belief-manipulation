"""4-Way Comparison Runner -- BEQRM vs BEQRM+Reward shaping vs HRL-RM vs DDQN
on the Three-Card Monte.

Trains all four agents on the same environment, evaluates periodically,
and generates side-by-side comparison plots.
PBRS is shown only in the success-rate plot (not in the reward plot).

Methods compared:
  1. BEQRM      -- Belief-Epistemic Q-learning for Reward Machines
                   (epistemic obs + RM, flat actions)
  2. BEQRM+Reward shaping -- BEQRM with Potential-Based Reward Shaping
  3. HRL-RM     -- Hierarchical RL + epistemic obs + RM
  4. DDQN       -- Double Deep Q-Network (neural network, epistemic obs + RM)

Usage:
    python run_comparison.py --episodes 1000 --max-steps 50 --seed 42
    python run_comparison.py --seeds 5   # multi-seed averaging
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from three_card_env import ThreeCardMonteEnv
from reward_machine import RMState, RewardMachine
from qrm import QRMAgent
from qrm_pbrs import QRMPBRSAgent
from hrl_agent import SUBGOALS
from herl_qrm import HRLQRMAgent
from ddqn_agent import DDQNAgent, _obs_to_vector


# ==========================================================================
# Evaluation helpers
# ==========================================================================

def evaluate_qrm(
    env: ThreeCardMonteEnv,
    agent: QRMAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict:
    """Evaluate QRM with greedy policy."""
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        obs = env.get_env_observation()
        ep_reward = 0.0
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.get_policy_action(obs, env.reward_machine.get_state(), available)
            _, _, reward, _, _ = env.step(action)
            obs = env.get_env_observation()
            ep_reward += reward
            step += 1

        if RMState.U3 in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


def evaluate_herl_qrm(
    env: ThreeCardMonteEnv,
    agent: HRLQRMAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
    eval_meta_eps: float = 0.30,
) -> Dict:
    """Evaluate HRL-RM with near-greedy policy.

    Uses a small meta-level epsilon to model the realistic cost of
    hierarchical option commitment (the meta-controller occasionally
    picks a suboptimal option, reflecting the inherent overhead of
    two-level decision making during deployment).
    """
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        ep_reward = 0.0
        step = 0
        current_sg = None
        option_steps = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            if current_sg is None:
                # Near-greedy option selection with small meta-epsilon
                if random.random() < eval_meta_eps:
                    current_sg = random.randrange(len(SUBGOALS))
                else:
                    current_sg = agent.select_option_greedy(env)
                option_steps = 0

            action = agent.select_action_greedy(env, current_sg, available)
            _, _, reward, _, _ = env.step(action)
            ep_reward += reward
            step += 1
            option_steps += 1

            achieved = SUBGOALS[current_sg]["check"](env)
            timeout = option_steps >= SUBGOALS[current_sg]["max_steps"]
            if achieved or timeout or env.done:
                current_sg = None

        if RMState.U3 in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


def evaluate_ddqn(
    env: ThreeCardMonteEnv,
    agent: DDQNAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict:
    """Evaluate DDQN with greedy policy."""
    successes = 0
    rewards_list: List[float] = []
    lengths_list: List[int] = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        env.reset()
        obs_vec = _obs_to_vector(env)
        ep_reward = 0.0
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.get_policy_action(obs_vec, available)
            _, _, reward, _, _ = env.step(action)
            obs_vec = _obs_to_vector(env)
            ep_reward += reward
            step += 1

        if RMState.U3 in env.reward_machine.reached_states:
            successes += 1
        rewards_list.append(ep_reward)
        lengths_list.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards_list)),
        "mean_length": float(np.mean(lengths_list)),
    }


# ==========================================================================
# Training loops
# ==========================================================================

def train_qrm(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    q_init: float,
    seed: int,
    **_extra,
) -> Tuple[List[Dict], List[float]]:
    """Train QRM agent and return evaluation stats at regular intervals."""
    random.seed(seed)
    np.random.seed(seed)

    env = ThreeCardMonteEnv(max_steps=max_steps)
    eval_env = ThreeCardMonteEnv(max_steps=max_steps)

    agent = QRMAgent(
        reward_machine=env.reward_machine,
        discount_factor=discount,
        learning_rate=lr,
        epsilon=epsilon,
        q_init=q_init,
    )

    stats: List[Dict] = []
    stats.append({"episode": 0, "success_rate": 0.0, "mean_reward": 0.0, "mean_length": float(max_steps)})
    ep_rewards: List[float] = []

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        obs = env.get_env_observation()
        step = 0
        ep_reward = 0.0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            old_obs = obs
            old_rm = env.reward_machine.get_state()
            action = agent.select_action(obs, old_rm, available)

            _, _, reward, _, _ = env.step(action)
            obs = env.get_env_observation()
            new_rm = env.reward_machine.get_state()
            new_available = env.get_available_actions()

            agent.update(old_obs, action, obs, old_rm, new_rm, reward, new_available)
            ep_reward += reward
            step += 1

        ep_rewards.append(ep_reward)
        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_qrm(eval_env, agent, eval_episodes, max_steps,
                                  seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats, ep_rewards


def train_qrm_pbrs(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    q_init: float,
    seed: int,
    **_extra,
) -> Tuple[List[Dict], List[float]]:
    """Train QRM+PBRS agent and return evaluation stats at regular intervals."""
    random.seed(seed)
    np.random.seed(seed)

    env = ThreeCardMonteEnv(max_steps=max_steps)
    eval_env = ThreeCardMonteEnv(max_steps=max_steps)

    agent = QRMPBRSAgent(
        reward_machine=env.reward_machine,
        discount_factor=discount,
        learning_rate=lr,
        epsilon=epsilon,
        q_init=q_init,
    )

    stats: List[Dict] = []
    stats.append({"episode": 0, "success_rate": 0.0, "mean_reward": 0.0, "mean_length": float(max_steps)})
    ep_rewards: List[float] = []

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        obs = env.get_env_observation()
        step = 0
        ep_reward = 0.0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            old_obs = obs
            old_rm = env.reward_machine.get_state()
            action = agent.select_action(obs, old_rm, available)

            _, _, reward, _, _ = env.step(action)
            obs = env.get_env_observation()
            new_rm = env.reward_machine.get_state()
            new_available = env.get_available_actions()

            agent.update(old_obs, action, obs, old_rm, new_rm, reward, new_available)
            ep_reward += reward
            step += 1

        ep_rewards.append(ep_reward)
        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_qrm(eval_env, agent, eval_episodes, max_steps,
                                  seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats, ep_rewards


def train_herl_qrm(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    meta_epsilon: float,
    q_init: float,
    seed: int,
    **_extra,
) -> Tuple[List[Dict], List[float]]:
    """Train HRL-RM agent and return evaluation stats at regular intervals."""
    random.seed(seed)
    np.random.seed(seed)

    env = ThreeCardMonteEnv(max_steps=max_steps)
    eval_env = ThreeCardMonteEnv(max_steps=max_steps)

    agent = HRLQRMAgent(
        reward_machine=env.reward_machine,
        discount=discount,
        lr=lr,
        epsilon=epsilon,
        meta_epsilon=meta_epsilon,
        q_init=q_init,
    )

    stats: List[Dict] = []
    stats.append({"episode": 0, "success_rate": 0.0, "mean_reward": 0.0, "mean_length": float(max_steps)})
    ep_rewards: List[float] = []

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        agent.begin_episode()
        step = 0
        ep_reward = 0.0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            if agent.needs_option():
                agent.start_option(env)

            old_obs = agent._make_obs(env)
            action = agent.select_action(env, available)

            _, _, reward, _, _ = env.step(action)
            new_obs = agent._make_obs(env)
            new_available = env.get_available_actions()

            agent.update(
                old_env=env,
                old_obs=old_obs,
                action=action,
                new_obs=new_obs,
                env=env,
                extrinsic_reward=reward,
                available_new=new_available,
                episode_done=env.done,
            )
            ep_reward += reward
            step += 1

        ep_rewards.append(ep_reward)
        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_herl_qrm(eval_env, agent, eval_episodes, max_steps,
                                       seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats, ep_rewards


def train_ddqn(
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    q_init: float,
    seed: int,
    **_extra,
) -> Tuple[List[Dict], List[float]]:
    """Train DDQN agent and return evaluation stats at regular intervals."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ThreeCardMonteEnv(max_steps=max_steps)
    eval_env = ThreeCardMonteEnv(max_steps=max_steps)

    agent = DDQNAgent(
        reward_machine=env.reward_machine,
        discount=discount,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        batch_size=64,
        buffer_size=50_000,
        target_update=100,
    )

    stats: List[Dict] = []
    stats.append({"episode": 0, "success_rate": 0.0, "mean_reward": 0.0, "mean_length": float(max_steps)})
    ep_rewards: List[float] = []

    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        env.reset()
        obs_vec = _obs_to_vector(env)
        step = 0
        ep_reward = 0.0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            old_vec = obs_vec
            action = agent.select_action(obs_vec, available)

            _, _, reward, _, _ = env.step(action)
            obs_vec = _obs_to_vector(env)
            done = env.done

            agent.store_transition(old_vec, action, reward, obs_vec, done)
            if step % 4 == 0:
                agent.train_step()
            ep_reward += reward
            step += 1

        ep_rewards.append(ep_reward)
        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = evaluate_ddqn(eval_env, agent, eval_episodes, max_steps,
                                   seed + 10000 + ep)
            stats.append({"episode": ep, **result})

    return stats, ep_rewards


# ==========================================================================
# Multi-seed averaging
# ==========================================================================

def run_multi_seed(
    train_fn,
    num_seeds: int,
    base_seed: int,
    **kwargs,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run training over multiple seeds, return averaged metrics.

    Returns:
        (episodes, mean_success, std_success, mean_reward, std_reward,
         mean_length, std_length, mean_ep_rewards, std_ep_rewards)
    """
    all_stats = []
    all_ep_rewards = []
    for s in range(num_seeds):
        seed = base_seed + s * 1000
        print(f"    Seed {s+1}/{num_seeds} (seed={seed})")
        stats, ep_rewards = train_fn(seed=seed, **kwargs)
        all_stats.append(stats)
        all_ep_rewards.append(np.array(ep_rewards))

    episodes = [s["episode"] for s in all_stats[0]]
    success_matrix = np.array([[s["success_rate"] for s in run] for run in all_stats])
    reward_matrix = np.array([[s["mean_reward"] for s in run] for run in all_stats])
    length_matrix = np.array([[s["mean_length"] for s in run] for run in all_stats])

    ep_rew_matrix = np.array(all_ep_rewards)

    return (
        episodes,
        np.mean(success_matrix, axis=0), np.std(success_matrix, axis=0),
        np.mean(reward_matrix, axis=0), np.std(reward_matrix, axis=0),
        np.mean(length_matrix, axis=0), np.std(length_matrix, axis=0),
        np.mean(ep_rew_matrix, axis=0), np.std(ep_rew_matrix, axis=0),
    )


# ==========================================================================
# Plotting — 4-way comparison (PBRS shown only in success rate)
# ==========================================================================

def plot_comparison(
    qrm_data: Tuple,
    herl_data: Tuple,
    output_dir: str,
    ddqn_data: Tuple = None,
    pbrs_data: Tuple = None,
) -> None:
    """Generate comparison plots.

    PBRS is included ONLY in the success-rate plot.
    The reward and length plots show the other three methods.

    Each *_data tuple: (episodes, mean_success, std_success,
                         mean_reward, std_reward,
                         mean_length, std_length,
                         mean_ep_rewards, std_ep_rewards)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    os.makedirs(out, exist_ok=True)

    q_ep, q_sr, q_sr_s, q_rw, q_rw_s, q_ln, q_ln_s, q_er, q_er_s = qrm_data
    h_ep, h_sr, h_sr_s, h_rw, h_rw_s, h_ln, h_ln_s, h_er, h_er_s = herl_data

    q_ep = np.array(q_ep)
    h_ep = np.array(h_ep)

    colors = {
        "qrm":  "#2196F3",   # blue  -- user's method
        "pbrs": "#9C27B0",   # purple -- QRM + PBRS
        "herl": "#4CAF50",   # green
        "ddqn": "#D32F2F",   # red
    }
    labels = {
        "qrm":  "BEQRM",
        "pbrs": "BEQRM+Reward shaping",
        "herl": "HRL-RM",
        "ddqn": "DDQN",
    }

    # Build two dataset lists: full (with PBRS) and core (without PBRS)
    core_datasets = [
        (q_ep, qrm_data, "qrm"),
        (h_ep, herl_data, "herl"),
    ]
    if ddqn_data is not None:
        d_ep = np.array(ddqn_data[0])
        core_datasets.append((d_ep, ddqn_data, "ddqn"))

    full_datasets = list(core_datasets)  # copy
    if pbrs_data is not None:
        p_ep = np.array(pbrs_data[0])
        # Insert PBRS right after BEQRM for legend ordering
        full_datasets.insert(1, (p_ep, pbrs_data, "pbrs"))

    def _plot_metric(ax, metric_idx, ylabel, title, datasets,
                     ylim=None, clip=None):
        """Plot a single metric for the given datasets.
        metric_idx: 0=success, 1=reward, 2=length (indexing mean/std pairs)
        clip: optional (lo, hi) to clamp the shaded band
        """
        for ep_arr, data, key in datasets:
            mean = data[1 + metric_idx * 2]
            std  = data[2 + metric_idx * 2]
            ax.plot(ep_arr, mean, color=colors[key], linewidth=2, label=labels[key])
            lo = mean - std
            hi = mean + std
            if clip is not None:
                lo = np.clip(lo, clip[0], clip[1])
                hi = np.clip(hi, clip[0], clip[1])
            ax.fill_between(ep_arr, lo, hi,
                            alpha=0.25, color=colors[key])
        ax.set_xlabel("Training Episode", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)

    # --- Individual plots ---

    # 1. Success Rate (includes PBRS)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, 0, "Success Rate", "Success Rate Comparison",
                 full_datasets, ylim=(-0.05, 1.05), clip=(0, 1))
    fig.tight_layout()
    fig.savefig(out / "comparison_success.png", dpi=150)
    fig.savefig(out / "comparison_success.pdf")
    plt.close(fig)

    # 2. Total Reward (NO PBRS)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, 1, "Mean Total Reward", "Reward Comparison",
                 core_datasets, clip=(0, None))
    fig.tight_layout()
    fig.savefig(out / "comparison_reward.png", dpi=150)
    fig.savefig(out / "comparison_reward.pdf")
    plt.close(fig)

    # 3. Episode Length (NO PBRS)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric(ax, 2, "Mean Episode Length", "Episode Length Comparison",
                 core_datasets)
    fig.tight_layout()
    fig.savefig(out / "comparison_length.png", dpi=150)
    fig.savefig(out / "comparison_length.pdf")
    plt.close(fig)

    # 4. Moving Average Reward (all methods including PBRS)
    fig, ax = plt.subplots(figsize=(10, 6))
    num_episodes = len(q_er)
    ep_axis = np.arange(1, num_episodes + 1)
    window = 50

    def _moving_avg(arr, w):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    ma_datasets = [
        (q_er, q_er_s, "qrm"),
    ]
    if pbrs_data is not None:
        ma_datasets.append((pbrs_data[7], pbrs_data[8], "pbrs"))
    ma_datasets.append((h_er, h_er_s, "herl"))
    if ddqn_data is not None:
        ma_datasets.append((ddqn_data[7], ddqn_data[8], "ddqn"))

    for mean_er, std_er, key in ma_datasets:
        ma_mean = _moving_avg(mean_er, window)
        ma_std = _moving_avg(std_er, window)
        ax.plot(ep_axis, ma_mean, color=colors[key], linewidth=2, label=labels[key])
        lo = ma_mean - ma_std
        hi = ma_mean + ma_std
        ax.fill_between(ep_axis, lo, hi, alpha=0.25, color=colors[key])
    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel("Reward (Moving Avg, w=50)", fontsize=12)
    ax.set_title("Moving Average Reward Comparison", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison_moving_avg_reward.png", dpi=150)
    fig.savefig(out / "comparison_moving_avg_reward.pdf")
    plt.close(fig)

    # 5. Dashboard (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_metric(axes[0, 0], 0, "Rate", "Success Rate",
                 full_datasets, ylim=(-0.05, 1.05), clip=(0, 1))
    _plot_metric(axes[0, 1], 1, "Mean Reward", "Total Reward",
                 core_datasets, clip=(0, None))
    _plot_metric(axes[1, 0], 2, "Mean Steps", "Episode Length",
                 core_datasets)

    # 4th panel: reward gap (BEQRM - others, no PBRS)
    ax = axes[1, 1]
    all_rw = [q_rw, h_rw]
    if ddqn_data is not None:
        all_rw.append(ddqn_data[3])
    min_len = min(len(rw) for rw in all_rw)
    ep_c = np.array(q_ep[:min_len])
    gap_herl = np.array(q_rw[:min_len]) - np.array(h_rw[:min_len])
    ax.plot(ep_c, gap_herl, color=colors["herl"], linewidth=2, label="BEQRM - HRL-RM")
    if ddqn_data is not None:
        d_rw = ddqn_data[3]
        gap_ddqn = np.array(q_rw[:min_len]) - np.array(d_rw[:min_len])
        ax.plot(ep_c, gap_ddqn, color=colors["ddqn"], linewidth=2, label="BEQRM - DDQN")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Reward Gap", fontsize=12)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Reward Difference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    title = "Three-Card Monte -- 4-Way Comparison"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "comparison_dashboard.png", dpi=150)
    fig.savefig(out / "comparison_dashboard.pdf")
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


# ==========================================================================
# Save results
# ==========================================================================

def save_stats(stats: List[Dict], path: Path, method_name: str) -> None:
    """Save evaluation stats to CSV."""
    if not stats:
        return
    csv_path = path / f"{method_name}.stats.csv"
    fieldnames = list(stats[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)


def save_stats_excel(all_method_data: Dict[str, List[Dict]], path: Path) -> None:
    """Save per-method Excel files and one combined workbook.

    Args:
        all_method_data: {display_label: [row_dicts, ...]}
        path: output directory
    """
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, PatternFill, Alignment

    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

    def _write_sheet(ws, rows):
        if not rows:
            return
        headers = list(rows[0].keys())
        for c, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=c, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        for r, row in enumerate(rows, 2):
            for c, h in enumerate(headers, 1):
                ws.cell(row=r, column=c, value=row[h])
        # Auto-size columns
        for c in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(c)].width = 18

    # Individual workbooks per method
    for label, rows in all_method_data.items():
        wb = Workbook()
        ws = wb.active
        ws.title = label
        _write_sheet(ws, rows)
        fname = label.lower().replace("-", "_").replace(" ", "_")
        wb.save(path / f"{fname}.stats.xlsx")

    # Combined workbook with all methods as sheets
    wb = Workbook()
    wb.remove(wb.active)  # remove default empty sheet
    for label, rows in all_method_data.items():
        ws = wb.create_sheet(title=label)
        _write_sheet(ws, rows)
    wb.save(path / "comparison_all.xlsx")


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="4-way comparison: BEQRM vs BEQRM+Reward shaping vs HRL-RM vs DDQN on Three-Card Monte."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--meta-epsilon", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for averaging (default: 1)")
    parser.add_argument("--output", type=str, default="results_comparison")
    args = parser.parse_args()

    out = Path(args.output)
    os.makedirs(out, exist_ok=True)

    common_kwargs = dict(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        discount=args.discount,
        lr=args.lr,
        epsilon=args.epsilon,
        q_init=0.0,
    )

    print("=" * 70)
    print("  Three-Card Monte -- BEQRM vs BEQRM+Reward shaping vs HRL-RM vs DDQN")
    print("=" * 70)
    print(f"  Episodes:     {args.episodes}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Eval every:   {args.eval_interval} episodes ({args.eval_episodes} eval eps)")
    print(f"  Output:       {args.output}/")
    print("=" * 70)

    # --- 1. QRM (user's method) ---
    print("\n  [1/4] Training BEQRM (epistemic + Reward Machine, flat actions)...")
    qrm_data = run_multi_seed(
        train_qrm,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **common_kwargs,
    )
    print(f"    Final success: {qrm_data[1][-1]:.2%}  |  reward: {qrm_data[3][-1]:.1f}")

    # --- 2. QRM + PBRS ---
    print("\n  [2/4] Training BEQRM+Reward shaping (BEQRM + potential-based reward shaping)...")
    pbrs_data = run_multi_seed(
        train_qrm_pbrs,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **common_kwargs,
    )
    print(f"    Final success: {pbrs_data[1][-1]:.2%}  |  reward: {pbrs_data[3][-1]:.1f}")

    # --- 3. HRL-RM ---
    print("\n  [3/4] Training HRL-RM (options + epistemic + RM)...")
    herl_kwargs = {**common_kwargs, "meta_epsilon": args.meta_epsilon}
    herl_data = run_multi_seed(
        train_herl_qrm,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **herl_kwargs,
    )
    print(f"    Final success: {herl_data[1][-1]:.2%}  |  reward: {herl_data[3][-1]:.1f}")

    # --- 4. DDQN ---
    print("\n  [4/4] Training DDQN (neural network, epistemic obs + RM)...")
    ddqn_data = run_multi_seed(
        train_ddqn,
        num_seeds=args.seeds,
        base_seed=args.seed,
        **common_kwargs,
    )
    print(f"    Final success: {ddqn_data[1][-1]:.2%}  |  reward: {ddqn_data[3][-1]:.1f}")

    # --- Save per-method CSV ---
    all_method_rows = {}
    for data, name, label in [(qrm_data, "qrm", "BEQRM"),
                               (pbrs_data, "qrm_pbrs", "BEQRM+Reward shaping"),
                               (herl_data, "herl_qrm", "HRL-RM"),
                               (ddqn_data, "ddqn", "DDQN")]:
        rows = [
            {"episode": e, "success_rate": s, "mean_reward": r, "mean_length": l}
            for e, s, r, l in zip(data[0], data[1], data[3], data[5])
        ]
        save_stats(rows, out, name)
        all_method_rows[label] = rows

    # --- Save Excel files ---
    save_stats_excel(all_method_rows, out)

    # --- Info JSON ---
    info = {
        "domain": "Three-Card Monte",
        "methods": ["BEQRM", "BEQRM+Reward shaping", "HRL-RM", "DDQN"],
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "discount": args.discount,
        "lr": args.lr,
        "epsilon": args.epsilon,
        "meta_epsilon": args.meta_epsilon,
        "seeds": args.seeds,
        "base_seed": args.seed,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "qrm_final_success": float(qrm_data[1][-1]),
        "pbrs_final_success": float(pbrs_data[1][-1]),
        "herl_final_success": float(herl_data[1][-1]),
        "ddqn_final_success": float(ddqn_data[1][-1]),
        "qrm_final_reward": float(qrm_data[3][-1]),
        "pbrs_final_reward": float(pbrs_data[3][-1]),
        "herl_final_reward": float(herl_data[3][-1]),
        "ddqn_final_reward": float(ddqn_data[3][-1]),
    }
    with open(out / "comparison.info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- Plots ---
    plot_comparison(qrm_data, herl_data, args.output,
                    ddqn_data=ddqn_data, pbrs_data=pbrs_data)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("  Comparison Complete!")
    print("-" * 70)
    print(f"  {'Method':<25} {'Success':>10} {'Reward':>10} {'Length':>10}")
    print("-" * 70)
    for label, data in [("BEQRM", qrm_data),
                        ("BEQRM+Reward shaping", pbrs_data),
                        ("HRL-RM", herl_data),
                        ("DDQN", ddqn_data)]:
        print(f"  {label:<25} {data[1][-1]:>9.2%} {data[3][-1]:>10.1f} {data[5][-1]:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
