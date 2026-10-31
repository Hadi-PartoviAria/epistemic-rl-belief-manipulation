"""Plot cumulative reward with moving average for Chop Cup comparison."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(__file__).parent / "results_comparison"

methods = {
    "BEQRM":      ("qrm.stats.csv",       "#2196F3"),
    "BEQRM+Reward shaping": ("qrm_pbrs.stats.csv",  "#9C27B0"),
    "HRL-RM":     ("herl_qrm.stats.csv",   "#4CAF50"),
    "HRL":        ("hrl.stats.csv",         "#FF9800"),
    "DDQN":       ("ddqn.stats.csv",        "#D32F2F"),
}

window = 10  # moving average window

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, (csv_file, color) in methods.items():
    df = pd.read_csv(results_dir / csv_file)
    episodes = df["episode"].values
    reward = df["mean_reward"].values

    # Cumulative reward
    cum_reward = np.cumsum(reward)
    ax1.plot(episodes, cum_reward, color=color, linewidth=2, label=name)

    # Moving average of mean reward
    ma = pd.Series(reward).rolling(window=window, min_periods=1).mean().values
    ax2.plot(episodes, ma, color=color, linewidth=2, label=name)

ax1.set_xlabel("Training Episode", fontsize=12)
ax1.set_ylabel("Cumulative Reward", fontsize=12)
ax1.set_title("Cumulative Reward", fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Training Episode", fontsize=12)
ax2.set_ylabel(f"Mean Reward (MA-{window})", fontsize=12)
ax2.set_title(f"Mean Reward (Moving Average, window={window})", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / "cumulative_reward_ma.png", dpi=150)
fig.savefig(results_dir / "cumulative_reward_ma.pdf")
plt.close(fig)
print(f"Saved to {results_dir / 'cumulative_reward_ma.pdf'}")
