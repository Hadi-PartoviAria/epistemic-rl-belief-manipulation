"""
Main Training Runner — QRM (Q-Learning for Reward Machines) on the Chop Cup.

Implements QRM from Icarte et al. (ICML 2018), following the reference:
  https://github.com/RodrigoToroIcarte/reward_machines

Training loop:
  1. ε-greedy exploration in each episode
  2. Online Q-update per step (current RM state only)
  3. Periodic greedy evaluation (success rate, reward)
  4. CSV output + plots

Usage:
    python main.py                          # Default: limes enabled
    python main.py --episodes 500 --seed 42 # Custom parameters
    python main.py --lr 0.1 --epsilon 0.1   # Tune learning rate / exploration
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from chop_cup_env import ChopCupEnv
from qrm import QRMAgent
from reward_machine import RMState


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Results from a batch of evaluation episodes."""
    success_rate: float           # fraction reaching terminal
    appear_rate: float            # fraction achieving "Appear" effect
    mean_reward: float            # average total reward
    mean_discounted_reward: float # average discounted reward
    mean_episode_length: float    # average steps per episode
    mean_rm_final: float          # average RM state index at end


def evaluate(
    env: ChopCupEnv,
    agent: QRMAgent,
    num_episodes: int = 100,
    max_steps: int = 50,
    seed: int = 0,
) -> EvalResult:
    """Evaluate the current greedy policy (no exploration)."""
    successes = 0
    appears = 0
    total_rewards = []
    discounted_rewards = []
    episode_lengths = []
    rm_finals = []

    for ep in range(num_episodes):
        random.seed(seed + ep)
        state, rm_state = env.reset()
        obs = env.get_env_observation()

        ep_reward = 0.0
        ep_disc_reward = 0.0
        step = 0

        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            action = agent.get_policy_action(obs, env.reward_machine.get_state(), available)
            state, rm_state, reward, done, info = env.step(action)
            obs = env.get_env_observation()

            ep_reward += reward
            ep_disc_reward += (agent.gamma ** step) * reward
            step += 1

        # Success = u_T was reached at least once during the episode
        rm_reached = env.reward_machine.reached_states
        if RMState.UT in rm_reached:
            successes += 1
        if RMState.U3 in rm_reached or RMState.UT in rm_reached:
            appears += 1

        rm_finals.append(env.reward_machine.get_state_index())
        total_rewards.append(ep_reward)
        discounted_rewards.append(ep_disc_reward)
        episode_lengths.append(step)

    n = max(num_episodes, 1)
    return EvalResult(
        success_rate=successes / n,
        appear_rate=appears / n,
        mean_reward=np.mean(total_rewards),
        mean_discounted_reward=np.mean(discounted_rewards),
        mean_episode_length=np.mean(episode_lengths),
        mean_rm_final=np.mean(rm_finals),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    num_episodes: int = 500,
    eval_interval: int = 1,
    eval_episodes: int = 100,
    max_steps: int = 50,
    discount_factor: float = 0.9,
    learning_rate: float = 0.1,
    epsilon: float = 0.1,
    q_init: float = 0.0,
    enable_limes: bool = False,
    max_limes: int = 3,
    seed: int = 0,
    output_dir: str = "results",
    verbose: bool = True,
) -> Tuple[QRMAgent, List[Dict]]:
    """Train the QRM agent on the Chop Cup trick.

    Uses online Q-learning for Reward Machines (QRM)
    following Icarte et al. (ICML 2018).

    Returns (trained_agent, stats_history).
    """
    random.seed(seed)
    np.random.seed(seed)

    env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)

    agent = QRMAgent(
        reward_machine=env.reward_machine,
        discount_factor=discount_factor,
        learning_rate=learning_rate,
        epsilon=epsilon,
        q_init=q_init,
    )

    os.makedirs(output_dir, exist_ok=True)
    stats_history: List[Dict] = []

    if verbose:
        print("=" * 70)
        print("  Chop Cup RL — QRM (Q-Learning for Reward Machines)")
        print("  Reference: Icarte et al., ICML 2018")
        print("=" * 70)
        print(f"  Episodes:        {num_episodes}")
        print(f"  Discount (γ):    {discount_factor}")
        print(f"  Learning rate:   {learning_rate}")
        print(f"  Epsilon (ε):     {epsilon}")
        print(f"  Q-init:          {q_init}")
        print(f"  Limes enabled:   {enable_limes}")
        print(f"  Max limes:       {max_limes}")
        print(f"  Seed:            {seed}")
        print(f"  Output:          {output_dir}/")
        print("=" * 70)

    # --- Evaluation before training (episode 0) ---
    eval_env = ChopCupEnv(enable_limes=enable_limes, max_limes=max_limes, max_steps=max_steps)
    eval_result = evaluate(eval_env, agent, num_episodes=eval_episodes,
                           max_steps=max_steps, seed=seed + 10000)
    stats = _make_stats_record(0, agent, eval_result)
    stats_history.append(stats)
    if verbose:
        _print_eval(0, eval_result, agent)

    # --- Training loop ---
    for ep in range(1, num_episodes + 1):
        random.seed(seed + ep)
        state, rm_state = env.reset()
        obs = env.get_env_observation()

        step = 0
        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break

            action = agent.select_action(obs, env.reward_machine.get_state(), available)
            old_obs = obs
            old_rm_state = env.reward_machine.get_state()

            state, rm_state, reward, done, info = env.step(action)
            obs = env.get_env_observation()
            new_available = env.get_available_actions()

            # QRM update: Q-update for the current RM state only
            agent.update(
                old_obs, action, obs,
                old_rm_state, env.reward_machine.get_state(),
                reward, new_available,
            )
            step += 1

        agent.training_episodes = ep

        # Periodic evaluation
        if ep % eval_interval == 0:
            eval_result = evaluate(eval_env, agent, num_episodes=eval_episodes,
                                   max_steps=max_steps, seed=seed + 10000 + ep)
            stats = _make_stats_record(ep, agent, eval_result)
            stats_history.append(stats)
            if verbose and (ep <= 10 or ep % max(1, num_episodes // 20) == 0 or ep == num_episodes):
                _print_eval(ep, eval_result, agent)

    # --- Save results ---
    _save_results(stats_history, agent, output_dir, enable_limes, num_episodes, seed,
                  discount_factor, learning_rate, epsilon, q_init, max_limes,
                  max_steps, eval_episodes)

    if verbose:
        print("\n" + "=" * 70)
        print("  Training Complete! (QRM)")
        print(f"  Final success rate: {stats_history[-1]['success_rate']:.2%}")
        print(f"  Final mean reward:  {stats_history[-1]['reward_t']:.4f}")
        print(f"  States explored:    {agent.num_states}")
        print(f"  Transitions stored: {agent.num_transitions}")
        print("=" * 70)
        print(f"\n  Q-Table:\n{agent.get_q_table_summary()}")
        print(f"\n  Results saved to {output_dir}/")

    return agent, stats_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats_record(episode: int, agent: QRMAgent, eval_result: EvalResult) -> Dict:
    return {
        "episode": episode,
        "observed_transitions": agent.num_transitions,
        "states_in_qtable": agent.num_states,
        "reward_d": eval_result.mean_discounted_reward,
        "reward_t": eval_result.mean_reward,
        "success_rate": eval_result.success_rate,
        "appear_rate": eval_result.appear_rate,
        "episode_length": eval_result.mean_episode_length,
    }


def _print_eval(episode: int, result: EvalResult, agent: QRMAgent) -> None:
    print(
        f"  Ep {episode:>5d} | "
        f"Success: {result.success_rate:5.2%} | "
        f"Appear: {result.appear_rate:5.2%} | "
        f"Reward: {result.mean_reward:6.3f} | "
        f"DiscR: {result.mean_discounted_reward:6.3f} | "
        f"Len: {result.mean_episode_length:5.1f} | "
        f"States: {agent.num_states:>3d} | "
        f"Trans: {agent.num_transitions:>4d}"
    )


def _save_results(
    stats_history: List[Dict],
    agent: QRMAgent,
    output_dir: str,
    enable_limes: bool,
    num_episodes: int,
    seed: int,
    discount_factor: float,
    learning_rate: float,
    epsilon: float,
    q_init: float,
    max_limes: int,
    max_steps: int,
    eval_episodes: int,
) -> None:
    """Save CSV stats, info JSON, and plots."""
    out = Path(output_dir)

    # --- CSV stats ---
    csv_path = out / "chop_cup.stats.csv"
    if stats_history:
        fieldnames = list(stats_history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_history)

    # --- Info JSON ---
    info = {
        "input": "chop_cup_rl (QRM — Icarte et al. 2018)",
        "algorithm": "QRM (Q-Learning for Reward Machines)",
        "seed": seed,
        "discount_factor": discount_factor,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "q_init": q_init,
        "num_episodes": num_episodes,
        "max_episode_len": max_steps,
        "enable_limes": enable_limes,
        "max_limes": max_limes,
        "evaluation_interval": 1,
        "evaluation_num_episodes": eval_episodes,
        "evaluation_max_episode_len": max_steps,
    }
    info_path = out / "chop_cup.info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # --- Plots ---
    try:
        _generate_plots(stats_history, output_dir, enable_limes)
    except ImportError:
        print("  (matplotlib not installed — skipping plots)")


def _generate_plots(stats_history: List[Dict], output_dir: str, enable_limes: bool) -> None:
    """Generate training curve plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def moving_average(data, window=25):
        """Compute moving average with given window size."""
        arr = np.array(data, dtype=float)
        if len(arr) < window:
            return arr
        cumsum = np.cumsum(arr)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        ma = cumsum[window - 1:] / window
        # Pad the beginning with expanding-window averages
        prefix = np.array([np.mean(arr[:i+1]) for i in range(window - 1)])
        return np.concatenate([prefix, ma])

    out = Path(output_dir)
    episodes = [s["episode"] for s in stats_history]
    success_rates = [s["success_rate"] for s in stats_history]
    appear_rates = [s["appear_rate"] for s in stats_history]
    rewards = [s["reward_t"] for s in stats_history]
    disc_rewards = [s["reward_d"] for s in stats_history]
    lengths = [s["episode_length"] for s in stats_history]

    rewards_ma = moving_average(rewards, window=50)
    disc_rewards_ma = moving_average(disc_rewards)

    # --- Plot 1: Success Rate ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(episodes, success_rates, "b-", linewidth=2, label="Terminal Success Rate")
    ax.plot(episodes, appear_rates, "r--", linewidth=1.5, label="Appear Rate")
    ax.set_xlabel("Training Episode", fontsize=13)
    ax.set_ylabel("Rate", fontsize=13)
    ax.set_title("Chop Cup RL — Learning Curve (Epistemic Deception)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "success_rate.png", dpi=150)
    fig.savefig(out / "success_rate.pdf")
    plt.close(fig)

    # --- Plot 2: Reward ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(episodes, rewards_ma, "g-", linewidth=2, label="Total Reward")
    ax.set_xlabel("Training Episode", fontsize=13)
    ax.set_ylabel("Average Reward", fontsize=13)
    ax.set_title("Chop Cup RL — Reward Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "reward_curve.png", dpi=150)
    fig.savefig(out / "reward_curve.pdf")
    plt.close(fig)

    # --- Plot 3: Episode Length ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(episodes, lengths, "k-", linewidth=2)
    ax.set_xlabel("Training Episode", fontsize=13)
    ax.set_ylabel("Average Episode Length", fontsize=13)
    ax.set_title("Chop Cup RL — Episode Length Over Training", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "episode_length.png", dpi=150)
    fig.savefig(out / "episode_length.pdf")
    plt.close(fig)

    # --- Plot 4: Combined Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(episodes, success_rates, "b-", linewidth=2)
    ax.set_title("Success Rate", fontsize=12)
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(episodes, rewards, "g-", linewidth=2, label="Total")
    ax.plot(episodes, disc_rewards, "m--", linewidth=1.5, label="Discounted")
    ax.set_title("Reward", fontsize=12)
    ax.set_ylabel("Avg Reward")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, lengths, "k-", linewidth=2)
    ax.set_title("Episode Length", fontsize=12)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Avg Steps")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, appear_rates, "r-", linewidth=2)
    ax.set_title("Appear Effect Rate", fontsize=12)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Chop Cup RL — Epistemic RL with Reward Machine", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "dashboard.png", dpi=150)
    fig.savefig(out / "dashboard.pdf")
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train QRM (Icarte et al. 2018) on the Chop Cup magic trick."
    )
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes (default: 500)")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Evaluate every N episodes (default: 1)")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Evaluation episodes per evaluation (default: 100)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode (default: 50)")
    parser.add_argument("--discount", type=float, default=0.9,
                        help="Discount factor γ (default: 0.9)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate α (default: 0.1)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Exploration rate ε (default: 0.1)")
    parser.add_argument("--q-init", type=float, default=0.0,
                        help="Q-value initialization (default: 0.0)")
    parser.add_argument("--enable-limes", action="store_true", default=True,
                        help="Enable lime finale phase (default: enabled)")
    parser.add_argument("--no-limes", dest="enable_limes", action="store_false",
                        help="Disable lime finale phase (ball-only mode)")
    parser.add_argument("--max-limes", type=int, default=3,
                        help="Maximum number of limes (default: 3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        discount_factor=args.discount,
        learning_rate=args.lr,
        epsilon=args.epsilon,
        q_init=args.q_init,
        enable_limes=args.enable_limes,
        max_limes=args.max_limes,
        seed=args.seed,
        output_dir=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
