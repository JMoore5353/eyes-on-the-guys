#!/usr/bin/env python3
"""Plot MCTS, Branch-and-Bound, and Forward Search: search depth vs. solve time and reward.

Usage:
    python3 plot_mcts_results.py [path/to/mcts_results.csv]

If no path is given, looks for mcts_results.csv in the current working directory.
The CSV is produced by the timing_and_reward_comparison executable.

Note: MCTS and BnB rewards use compute_reward_model (delta shared-info norm).
      ForwardSearch uses its own reward function (full shared-info matrix norm).
      Rewards across algorithms are therefore NOT directly comparable.
"""

import sys
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

ALGO_LABELS = {
    "mcts": "MCTS",
    "bnb":  "Branch & Bound",
    "fs":   "Forward Search",
}
ALGO_COLORS = {
    "mcts": "tab:blue",
    "bnb":  "tab:orange",
    "fs":   "tab:green",
}


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"algorithm", "depth", "trial", "time_ms", "reward"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV is missing columns. Expected: {required}, got: {set(df.columns)}")
    return df


def plot_results(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for algo, group in df.groupby("algorithm"):
        stats = group.groupby("depth").agg(
            time_mean=("time_ms", "mean"),
            time_std=("time_ms", "std"),
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
        ).reset_index()

        label = ALGO_LABELS.get(algo, algo)
        color = ALGO_COLORS.get(algo, None)

        axes[0].errorbar(
            stats["depth"], stats["time_mean"],
            yerr=stats["time_std"],
            label=label, color=color,
            marker="o", capsize=4, linewidth=1.5,
        )

        axes[1].errorbar(
            stats["depth"], stats["reward_mean"],
            yerr=stats["reward_std"],
            label=label, color=color,
            marker="o", capsize=4, linewidth=1.5,
        )

    axes[0].set_xlabel("Search Depth")
    axes[0].set_ylabel("Solve Time (ms)")
    axes[0].set_title("Search Depth vs. Solve Time")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_xlabel("Search Depth")
    axes[1].set_ylabel("Algorithm Reward")
    axes[1].set_title(
        "Search Depth vs. Reward\n"
        "(MCTS/BnB and ForwardSearch rewards are not directly comparable)"
    )
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = output_dir / "timing_reward_depth_results.png"
    # fig.savefig(out_path, dpi=150)
    # print(f"Plot saved to: {out_path}")
    plt.show()


def main():
    csv_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("timing_and_reward_results.csv")
    if not csv_path.exists():
        print(f"Error: file not found: {csv_path}")
        sys.exit(1)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df.groupby(["algorithm", "depth"])[["time_ms", "reward"]].mean().to_string())

    plot_results(df, output_dir=csv_path.parent)


if __name__ == "__main__":
    main()
