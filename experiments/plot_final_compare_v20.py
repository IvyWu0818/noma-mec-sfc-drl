"""
plot_final_compare_v20.py
Final 5-way comparison: SAC (V20, primary model) vs TD3 (V17) vs PPO (V17) vs
Greedy vs GA -- 6-panel bar comparison (reward, delay, delay decomposition,
cpu_viol_rate, channel overflow, timeout) plus a log-scale runtime panel.

Run: python -m experiments.plot_final_compare_v20
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.baselines_v17 import print_summary

EVAL_FILE   = "results/final_compare_v20.json"
OUTPUT_DIR  = "results/figures_final_compare"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_compare_v20.png")

ALGOS     = ["SAC", "TD3", "PPO", "Greedy", "GA"]
DRL_ALGOS = ["SAC", "TD3", "PPO"]
COLORS = {"SAC": "#7E57C2", "TD3": "#2E86AB", "PPO": "#82C882",
          "Greedy": "#E07B54", "GA": "#C2A83E"}


def savefig_both(fig, png_path, **kwargs):
    """Save fig as PNG and as a sibling .svg (vector) file."""
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg", **kwargs)


def mean_std(data, algo, key):
    vals = data.get(algo, {}).get(key, [])
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


def bar_panel(ax, data, key, title, ylabel, algos=None):
    algos = algos or ALGOS
    means = [mean_std(data, a, key)[0] for a in algos]
    stds  = [mean_std(data, a, key)[1] for a in algos]
    bars = ax.bar(algos, means, yerr=stds, capsize=4,
                   color=[COLORS[a] for a in algos], alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)


DECOMP_KEYS   = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
DECOMP_LABELS = ["t_ul (upload)", "t_comp (compute)", "t_link (backhaul)"]
DECOMP_COLORS = ["#7DC3E8", "#F0A070", "#82C882"]


def decomp_panel(ax, data):
    bottoms = np.zeros(len(ALGOS))
    for key, label, color in zip(DECOMP_KEYS, DECOMP_LABELS, DECOMP_COLORS):
        means = np.array([mean_std(data, a, key)[0] for a in ALGOS])
        ax.bar(ALGOS, means, bottom=bottoms, color=color, alpha=0.85, label=label)
        bottoms += means
    ax.set_title("Delay Decomposition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Delay (ms)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def runtime_panel(ax, data):
    means = [mean_std(data, a, "episode_runtime_sec")[0] for a in ALGOS]
    stds  = [mean_std(data, a, "episode_runtime_sec")[1] for a in ALGOS]
    ax.bar(ALGOS, means, yerr=stds, capsize=4,
           color=[COLORS[a] for a in ALGOS], alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Per-Episode Runtime (100 tasks, CPU)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Seconds (log scale)", fontsize=9)
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)


# (key, title, ylabel, filename) for the single-metric bar panels
SINGLE_CFG = [
    ("episode_rewards",                "Cumulative Reward",       "Reward",            "final_compare_v20_reward.png"),
    ("episode_avg_delay",              "End-to-End Delay (ms)",   "Avg Delay (ms)",    "final_compare_v20_delay.png"),
    ("episode_cpu_viol_rate",          "CPU Violation Rate",       "Rate (raw / 135)",  "final_compare_v20_cpu_viol.png"),
    ("episode_channel_overflow_ratio", "Channel Overflow Ratio",   "Ratio",             "final_compare_v20_ch_overflow.png"),
    ("episode_timeout_ratio",          "Timeout Ratio",            "Ratio",             "final_compare_v20_timeout.png"),
]


def main():
    if not os.path.exists(EVAL_FILE):
        print(f"Missing {EVAL_FILE} -- run experiments.eval_final_compare_v20 first")
        return

    with open(EVAL_FILE) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 0. Reward: DRL-only (SAC vs TD3 vs PPO, no Greedy/GA) ───────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_panel(ax, data, "episode_rewards", "Cumulative Reward (DRL only)",
              "Reward", algos=DRL_ALGOS)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "final_compare_v20_reward_drl.png"))
    plt.close(fig)
    print("saved final_compare_v20_reward_drl.png (+ .svg)")

    # ── 1. Individual eval plots ────────────────────────────────────────────
    for key, title, ylabel, fname in SINGLE_CFG:
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_panel(ax, data, key, title, ylabel)
        fig.tight_layout()
        savefig_both(fig, os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)
        print(f"saved {fname} (+ .svg)")

    fig, ax = plt.subplots(figsize=(8, 5))
    decomp_panel(ax, data)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "final_compare_v20_delay_decomp.png"))
    plt.close(fig)
    print("saved final_compare_v20_delay_decomp.png (+ .svg)")

    fig, ax = plt.subplots(figsize=(8, 5))
    runtime_panel(ax, data)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "final_compare_v20_runtime.png"))
    plt.close(fig)
    print("saved final_compare_v20_runtime.png (+ .svg)")

    # ── 2. 7-panel overview ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.32)

    bar_panel(fig.add_subplot(gs[0, 0]), data, "episode_rewards",
              "Cumulative Reward", "Reward")
    bar_panel(fig.add_subplot(gs[0, 1]), data, "episode_avg_delay",
              "End-to-End Delay (ms)", "Avg Delay (ms)")
    decomp_panel(fig.add_subplot(gs[0, 2]), data)

    bar_panel(fig.add_subplot(gs[1, 0]), data, "episode_cpu_viol_rate",
              "CPU Violation Rate", "Rate (raw / 135)")
    bar_panel(fig.add_subplot(gs[1, 1]), data, "episode_channel_overflow_ratio",
              "Channel Overflow Ratio", "Ratio")
    bar_panel(fig.add_subplot(gs[1, 2]), data, "episode_timeout_ratio",
              "Timeout Ratio", "Ratio")

    fig.suptitle(
        "SAC (V20, primary) vs TD3 vs PPO vs Greedy vs GA -- Evaluation Comparison\n"
        f"({len(data['SAC']['episode_rewards'])} eval episodes x 100 tasks, "
        "error bars = std across episodes)",
        fontsize=13, fontweight="bold", y=1.02
    )
    savefig_both(fig, OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_FILE} (+ .svg)")

    n = len(data['SAC']['episode_rewards'])
    print(f"\n--- Final comparison: {n} episodes x 100 tasks (mean +/- std) ---")
    print_summary(data)


if __name__ == "__main__":
    main()
