"""Ablation runner -- Autonomous Driving (7x7 Waymo grid, Case Study 3).

Trains five configurations of the ablation table on the same 7x7
Waymo-style traffic environment using the configurable AblationAgent.
Each variant is run for `--seeds` seeds and we report:

    final success, final reward, AUC of success curve,
    convergence episode (first ep where rolling success >= threshold),
    95% CI around the final success, mean SAT cost (#queries + wall sec).

Plots produced:

    ablation_success.{png,pdf}    -- success-rate curves, all variants
    ablation_central.{png,pdf}    -- the paper 4-curve subset
    ablation_dashboard.{png,pdf}  -- success / reward / length / Q-states
    ablation_table.{csv,md}       -- final summary table

Usage:
    python run_ablation.py --episodes 1000 --max-steps 150 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from autonomous_driving_env import AutonomousDrivingEnv


def _smooth(y, alpha=0.65):
    out = np.array(y, dtype=float).copy()
    for i in range(1, len(out)):
        out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]
    return out
from reward_machine import RMState
from ablation_agent import (
    AblationAgent, VARIANTS, LABELS, COLORS, CENTRAL_FIGURE_VARIANTS,
)


def _evaluate(
    env: AutonomousDrivingEnv,
    agent: AblationAgent,
    num_episodes: int,
    max_steps: int,
    seed: int,
) -> Dict[str, float]:
    successes = 0
    rewards: List[float] = []
    lengths: List[int] = []

    for ep in range(num_episodes):
        env.reset()
        rm_state = env.reward_machine.get_state()
        obs = agent.make_obs(env, rm_state)
        ep_reward = 0.0
        step = 0
        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.get_policy_action(obs, available)
            _, rm_state, reward, _, _ = env.step(action)
            obs = agent.make_obs(env, rm_state)
            ep_reward += reward
            step += 1

        if RMState.UT in env.reward_machine.reached_states:
            successes += 1
        rewards.append(ep_reward)
        lengths.append(step)

    n = max(num_episodes, 1)
    return {
        "success_rate": successes / n,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def _train_variant(
    variant_key: str,
    num_episodes: int,
    max_steps: int,
    eval_interval: int,
    eval_episodes: int,
    discount: float,
    lr: float,
    epsilon: float,
    q_init: float,
    seed: int,
) -> Tuple[List[Dict], List[float], Dict[str, float]]:
    random.seed(seed)
    np.random.seed(seed)

    env = AutonomousDrivingEnv(max_steps=max_steps, seed=seed)
    eval_env = AutonomousDrivingEnv(max_steps=max_steps, seed=seed + 7919)

    flags = VARIANTS[variant_key]
    agent = AblationAgent(
        reward_machine=env.reward_machine,
        discount_factor=discount,
        learning_rate=lr,
        epsilon=epsilon,
        q_init=q_init,
        **flags,
    )

    stats: List[Dict] = [{"episode": 0, "success_rate": 0.0,
                          "mean_reward": 0.0, "mean_length": float(max_steps)}]
    ep_rewards: List[float] = []

    t0 = time.perf_counter()
    for ep in range(1, num_episodes + 1):
        env.reset()
        rm_state = env.reward_machine.get_state()
        obs = agent.make_obs(env, rm_state)
        step = 0
        ep_reward = 0.0
        while not env.done and step < max_steps:
            available = env.get_available_actions()
            if not available:
                break
            old_obs = obs
            old_rm = rm_state
            action = agent.select_action(obs, available)

            _, rm_state, reward, _, _ = env.step(action)
            obs = agent.make_obs(env, rm_state)
            new_available = env.get_available_actions()

            agent.update(old_obs, action, obs, old_rm, rm_state, reward,
                         new_available)
            ep_reward += reward
            step += 1

        ep_rewards.append(ep_reward)
        agent.training_episodes = ep

        if ep % eval_interval == 0:
            result = _evaluate(eval_env, agent, eval_episodes, max_steps,
                               seed + 10_000 + ep)
            stats.append({"episode": ep, **result})

    train_wall_s = time.perf_counter() - t0

    cost = {
        "sat_query_count": agent.sat_query_count,
        "sat_query_time_s": agent.sat_query_time_s,
        "train_wall_s": train_wall_s,
        "final_q_states": agent.num_states,
    }
    return stats, ep_rewards, cost


def _run_multi_seed(
    variant_key: str,
    num_seeds: int,
    base_seed: int,
    **train_kwargs,
) -> Dict:
    all_stats: List[List[Dict]] = []
    all_ep_rewards: List[np.ndarray] = []
    costs: List[Dict[str, float]] = []
    for s in range(num_seeds):
        seed = base_seed + s * 100_000
        print(f"    Seed {s + 1}/{num_seeds} (seed={seed})")
        stats, ep_rewards, cost = _train_variant(
            variant_key=variant_key, seed=seed, **train_kwargs,
        )
        all_stats.append(stats)
        all_ep_rewards.append(np.array(ep_rewards))
        costs.append(cost)

    episodes = [s["episode"] for s in all_stats[0]]
    success_matrix = np.array([[s["success_rate"] for s in run] for run in all_stats])
    reward_matrix = np.array([[s["mean_reward"] for s in run] for run in all_stats])
    length_matrix = np.array([[s["mean_length"] for s in run] for run in all_stats])
    ep_rew_matrix = np.array(all_ep_rewards)

    return {
        "variant": variant_key,
        "episodes": episodes,
        "success_mean": np.mean(success_matrix, axis=0),
        "success_std":  np.std(success_matrix, axis=0),
        "success_all":  success_matrix,
        "reward_mean":  np.mean(reward_matrix, axis=0),
        "reward_std":   np.std(reward_matrix, axis=0),
        "reward_all":   reward_matrix,
        "length_mean":  np.mean(length_matrix, axis=0),
        "length_std":   np.std(length_matrix, axis=0),
        "length_all":   length_matrix,
        "ep_reward_mean": np.mean(ep_rew_matrix, axis=0),
        "ep_reward_std":  np.std(ep_rew_matrix, axis=0),
        "costs": costs,
    }


# ---------------------------------------------------------------------------
# Summary metrics (table)
# ---------------------------------------------------------------------------

def _ci95(values: np.ndarray) -> Tuple[float, float]:
    """Return (lo, hi) for the 95% CI of the mean over the seed axis."""
    n = len(values)
    if n <= 1:
        m = float(np.mean(values))
        return m, m
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1)) / math.sqrt(n)
    half = 1.96 * sem
    return mean - half, mean + half


def _auc(episodes: List[int], curve: np.ndarray) -> float:
    """Trapezoidal area under the success-rate curve, normalised to [0, 1]."""
    eps = np.asarray(episodes, dtype=float)
    if eps[-1] == eps[0]:
        return float(curve[-1])
    area = np.trapz(curve, eps)
    return float(area / (eps[-1] - eps[0]))


def _convergence_episode(
    episodes: List[int], curve: np.ndarray, threshold: float = 0.9,
    window: int = 3,
) -> int | None:
    """First episode where rolling-mean success >= threshold for `window` checkpoints."""
    if len(curve) < window:
        return None
    for i in range(window - 1, len(curve)):
        if float(np.mean(curve[i - window + 1: i + 1])) >= threshold:
            return int(episodes[i])
    return None


def _summary_row(data: Dict) -> Dict[str, float]:
    final_success_per_seed = data["success_all"][:, -1]
    lo, hi = _ci95(final_success_per_seed)
    auc = _auc(data["episodes"], data["success_mean"])
    conv = _convergence_episode(data["episodes"], data["success_mean"])
    sat_count = float(np.mean([c["sat_query_count"] for c in data["costs"]]))
    sat_time  = float(np.mean([c["sat_query_time_s"] for c in data["costs"]]))
    train_s   = float(np.mean([c["train_wall_s"] for c in data["costs"]]))
    return {
        "variant": data["variant"],
        "label": LABELS[data["variant"]],
        "final_success": float(data["success_mean"][-1]),
        "final_success_ci_lo": lo,
        "final_success_ci_hi": hi,
        "final_reward": float(data["reward_mean"][-1]),
        "auc_success": auc,
        "convergence_ep": conv if conv is not None else "n/a",
        "mean_sat_queries": sat_count,
        "mean_sat_seconds": sat_time,
        "mean_train_seconds": train_s,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_curves(
    results: Dict[str, Dict],
    keys: List[str],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    ylim: Tuple[float, float] | None = None,
    clip: Tuple[float | None, float | None] | None = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in keys:
        data = results[key]
        ep = np.array(data["episodes"])
        all_key = f"{metric}_all"
        if all_key in data:
            smoothed = np.array([_smooth(row) for row in data[all_key]])
            mean = np.mean(smoothed, axis=0)
            std  = np.std(smoothed, axis=0)
        else:
            mean = _smooth(data[f"{metric}_mean"])
            std  = _smooth(data[f"{metric}_std"])
        ax.plot(ep, mean, color=COLORS[key], linewidth=2, label=LABELS[key])
        lo = mean - std
        hi = mean + std
        if clip is not None:
            lo = np.clip(lo, clip[0], clip[1])
            hi = np.clip(hi, clip[0], clip[1])
        ax.fill_between(ep, lo, hi, alpha=0.20, color=COLORS[key])
    ax.set_xlabel("Training Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_dashboard(results: Dict[str, Dict], keys: List[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def panel(ax, metric, ylabel, title, ylim=None, clip=None):
        for key in keys:
            data = results[key]
            ep = np.array(data["episodes"])
            all_key = f"{metric}_all"
            if all_key in data:
                smoothed = np.array([_smooth(row) for row in data[all_key]])
                mean = np.mean(smoothed, axis=0)
                std  = np.std(smoothed, axis=0)
            else:
                mean = _smooth(data[f"{metric}_mean"])
                std  = _smooth(data[f"{metric}_std"])
            ax.plot(ep, mean, color=COLORS[key], linewidth=2, label=LABELS[key])
            lo = mean - std
            hi = mean + std
            if clip is not None:
                lo = np.clip(lo, clip[0], clip[1])
                hi = np.clip(hi, clip[0], clip[1])
            ax.fill_between(ep, lo, hi, alpha=0.18, color=COLORS[key])
        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    panel(axes[0, 0], "success", "Success Rate", "Success Rate",
          ylim=(-0.05, 1.05), clip=(0.0, 1.0))
    panel(axes[0, 1], "reward",  "Mean Reward",  "Total Reward",
          clip=(0.0, None))
    panel(axes[1, 0], "length",  "Mean Steps",   "Episode Length")

    ax = axes[1, 1]
    # Q-table growth (one bar per variant). For PBRS we count training
    # episodes implicitly via convergence -- here we just show |Q|.
    labels = [LABELS[k] for k in keys]
    sizes = [float(np.mean([c["final_q_states"] for c in results[k]["costs"]]))
             for k in keys]
    bar_colors = [COLORS[k] for k in keys]
    ax.barh(labels, sizes, color=bar_colors)
    ax.set_xlabel("# tabular states learned (avg over seeds)")
    ax.set_title("Effective state-space size")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Autonomous Driving (7x7 Waymo grid) -- BEQRM ablation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def _save_per_variant_csv(results: Dict[str, Dict], outdir: Path) -> None:
    for key, data in results.items():
        rows = [
            {"episode": e,
             "success_mean": float(sm), "success_std": float(ss),
             "reward_mean":  float(rm), "reward_std":  float(rs),
             "length_mean":  float(lm), "length_std":  float(ls)}
            for e, sm, ss, rm, rs, lm, ls in zip(
                data["episodes"],
                data["success_mean"], data["success_std"],
                data["reward_mean"],  data["reward_std"],
                data["length_mean"],  data["length_std"],
            )
        ]
        with open(outdir / f"{key}.stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _save_table(summary: List[Dict], outdir: Path) -> None:
    csv_path = outdir / "ablation_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    md_path = outdir / "ablation_table.md"
    headers = ["Variant", "Final success (95% CI)", "Final reward",
               "AUC", "Conv. ep (>=0.9)", "SAT queries",
               "SAT sec", "Train sec"]
    with open(md_path, "w") as f:
        f.write("# BEQRM Ablation -- Autonomous Driving (7x7 Waymo grid)\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in summary:
            ci = f"{row['final_success']:.2%}  [{row['final_success_ci_lo']:.2%},"\
                 f" {row['final_success_ci_hi']:.2%}]"
            f.write(
                f"| {row['label']} "
                f"| {ci} "
                f"| {row['final_reward']:.3f} "
                f"| {row['auc_success']:.3f} "
                f"| {row['convergence_ep']} "
                f"| {row['mean_sat_queries']:.0f} "
                f"| {row['mean_sat_seconds']:.2f} "
                f"| {row['mean_train_seconds']:.1f} |\n"
            )


def _save_info(args, summary, outdir: Path) -> None:
    info = {
        "domain": "Autonomous Driving (7x7 Waymo-style traffic, epistemic ambiguity) -- BEQRM ablation",
        "config": vars(args),
        "variants": [row["label"] for row in summary],
        "summary": summary,
    }
    with open(outdir / "ablation.info.json", "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BEQRM ablation runner (7x7 Waymo Autonomous Driving).",
    )
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--q-init", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="ablation_results")
    parser.add_argument(
        "--variants", type=str, default="",
        help="Comma-separated subset of variant keys to run (default: all).",
    )
    args = parser.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    selected = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants else list(VARIANTS.keys())
    )
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. "
                         f"Choices: {list(VARIANTS)}")

    common = dict(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        discount=args.discount,
        lr=args.lr,
        epsilon=args.epsilon,
        q_init=args.q_init,
    )

    print("=" * 72)
    print("  BEQRM Ablation -- Autonomous Driving (7x7 Waymo, epistemic ambiguity)")
    print("=" * 72)
    print(f"  Variants : {selected}")
    print(f"  Episodes : {args.episodes}, max-steps {args.max_steps}, "
          f"seeds {args.seeds}")
    print(f"  Eval     : every {args.eval_interval} ep ({args.eval_episodes} eval-eps)")
    print(f"  Hyperp   : gamma={args.discount} lr={args.lr} eps={args.epsilon} "
          f"q_init={args.q_init}")
    print(f"  Output   : {outdir}/")
    print("=" * 72)

    results: Dict[str, Dict] = {}
    for i, key in enumerate(selected, 1):
        print(f"\n  [{i}/{len(selected)}] Variant: {LABELS[key]}  ({key})")
        results[key] = _run_multi_seed(
            variant_key=key,
            num_seeds=args.seeds,
            base_seed=args.seed,
            **common,
        )
        sm = results[key]["success_mean"][-1]
        rm = results[key]["reward_mean"][-1]
        print(f"    Final success: {sm:.2%}  |  reward: {rm:.3f}")

    summary = [_summary_row(results[k]) for k in selected]
    _save_per_variant_csv(results, outdir)
    _save_table(summary, outdir)
    _save_info(args, summary, outdir)

    # Plot: all variants, success
    _plot_curves(
        results, selected, metric="success",
        ylabel="Success Rate",
        title="Success Rate -- all ablation variants",
        out_path=outdir / "ablation_success",
        ylim=(-0.05, 1.05), clip=(0.0, 1.0),
    )

    # Plot: central paper figure (5 curves)
    central_keys = [k for k in CENTRAL_FIGURE_VARIANTS if k in selected]
    if len(central_keys) >= 2:
        _plot_curves(
            results, central_keys, metric="success",
            ylabel="Success Rate",
            title="Central ablation -- BEQRM vs. ingredient removals",
            out_path=outdir / "ablation_central",
            ylim=(-0.05, 1.05), clip=(0.0, 1.0),
        )

    _plot_dashboard(results, selected, outdir / "ablation_dashboard")

    print("\n" + "=" * 72)
    print("  Ablation complete!")
    print("-" * 72)
    print(f"  {'Variant':<22} {'Success':>16} {'Reward':>9} {'AUC':>6} "
          f"{'Conv@0.9':>10} {'SAT q':>9}")
    print("-" * 72)
    for row in summary:
        ci = f"[{row['final_success_ci_lo']:.2f},{row['final_success_ci_hi']:.2f}]"
        print(f"  {row['label']:<22} {row['final_success']:>7.2%} {ci:>8} "
              f"{row['final_reward']:>9.3f} {row['auc_success']:>6.3f} "
              f"{str(row['convergence_ep']):>10} {row['mean_sat_queries']:>9.0f}")
    print("=" * 72)
    print(f"  Artefacts written to {outdir}/")


if __name__ == "__main__":
    main()
