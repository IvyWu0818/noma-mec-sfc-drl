"""
plot_drl_compare_v17.py
TD3 vs DDPG vs SAC vs PPO on IIoTEnvV17 -- training-curve comparison plus a
6-panel deterministic-eval bar comparison (reward, delay, delay decomposition,
cpu_viol_rate, channel overflow, timeout).

Run: python -m experiments.plot_drl_compare_v17
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.baselines_v17 import print_summary

TRAIN_FILES = {
    "TD3":  "results/td3_v17_training_metrics.json",
    "DDPG": "results/ddpg_v17_training_metrics.json",
    "SAC":  "results/sac_v17_training_metrics.json",
    "PPO":  "results/ppo_v17_training_metrics.json",
}
EVAL_FILE  = "results/drl_eval_v17.json"
OUTPUT_DIR = "results/figures_drl_compare"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "drl_compare_v17.png")

ALGOS  = ["TD3", "DDPG", "SAC", "PPO"]
COLORS = {"TD3": "#2E86AB", "DDPG": "#E07B54", "SAC": "#7E57C2", "PPO": "#82C882"}


def savefig_both(fig, png_path, **kwargs):
    """Save fig as PNG and as a sibling .svg (vector) file."""
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg", **kwargs)


def smooth(x, window=100):
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def training_curve_panel(ax):
    for algo in ALGOS:
        with open(TRAIN_FILES[algo]) as f:
            m = json.load(f)
        rewards = m["episode_rewards"]
        smoothed = smooth(rewards, window=100)
        x = np.arange(len(smoothed)) + 100
        ax.plot(x, smoothed, label=algo, color=COLORS[algo], linewidth=1.5)
    ax.set_title("Training Reward (100-episode moving avg)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)


def mean_std(data, algo, key):
    vals = data.get(algo, {}).get(key, [])
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


def bar_panel(ax, data, key, title, ylabel):
    means = [mean_std(data, a, key)[0] for a in ALGOS]
    stds  = [mean_std(data, a, key)[1] for a in ALGOS]
    bars = ax.bar(ALGOS, means, yerr=stds, capsize=4,
                   color=[COLORS[a] for a in ALGOS], alpha=0.85)
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


# (key, title, ylabel, filename) for the single-metric bar panels
SINGLE_CFG = [
    ("episode_rewards",                "Cumulative Reward",       "Reward",            "drl_compare_v17_reward.png"),
    ("episode_avg_delay",              "End-to-End Delay (ms)",   "Avg Delay (ms)",    "drl_compare_v17_delay.png"),
    ("episode_cpu_viol_rate",          "CPU Violation Rate",       "Rate (raw / 135)",  "drl_compare_v17_cpu_viol.png"),
    ("episode_channel_overflow_ratio", "Channel Overflow Ratio",   "Ratio",             "drl_compare_v17_ch_overflow.png"),
    ("episode_timeout_ratio",          "Timeout Ratio",            "Ratio",             "drl_compare_v17_timeout.png"),
]


def main():
    for algo, path in TRAIN_FILES.items():
        if not os.path.exists(path):
            print(f"Missing {path} -- train {algo} first (agents.train_{algo.lower()}_v17)")
            return
    if not os.path.exists(EVAL_FILE):
        print(f"Missing {EVAL_FILE} -- run experiments.eval_drl_compare_v17 first")
        return

    with open(EVAL_FILE) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Training curve comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    training_curve_panel(ax)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "drl_compare_v17_training_curve.png"))
    plt.close(fig)
    print("saved drl_compare_v17_training_curve.png (+ .svg)")

    # ── 2. Individual eval plots ────────────────────────────────────────────
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
    savefig_both(fig, os.path.join(OUTPUT_DIR, "drl_compare_v17_delay_decomp.png"))
    plt.close(fig)
    print("saved drl_compare_v17_delay_decomp.png (+ .svg)")

    # ── 3. 7-panel overview (training curve + 6 eval bars) ─────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.32)

    training_curve_panel(fig.add_subplot(gs[0, :]))

    bar_panel(fig.add_subplot(gs[1, 0]), data, "episode_rewards",
              "Cumulative Reward", "Reward")
    bar_panel(fig.add_subplot(gs[1, 1]), data, "episode_avg_delay",
              "End-to-End Delay (ms)", "Avg Delay (ms)")
    decomp_panel(fig.add_subplot(gs[1, 2]), data)

    bar_panel(fig.add_subplot(gs[2, 0]), data, "episode_cpu_viol_rate",
              "CPU Violation Rate", "Rate (raw / 135)")
    bar_panel(fig.add_subplot(gs[2, 1]), data, "episode_channel_overflow_ratio",
              "Channel Overflow Ratio", "Ratio")
    bar_panel(fig.add_subplot(gs[2, 2]), data, "episode_timeout_ratio",
              "Timeout Ratio", "Ratio")

    fig.suptitle(
        "TD3 vs DDPG vs SAC vs PPO -- Training & Evaluation Comparison\n"
        f"(1M timesteps training; {len(data['TD3']['episode_rewards'])} eval "
        "episodes x 100 tasks, error bars = std across episodes)",
        fontsize=13, fontweight="bold", y=1.01
    )
    savefig_both(fig, OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_FILE} (+ .svg)")

    n = len(data['TD3']['episode_rewards'])
    print(f"\n--- DRL comparison: {n} episodes x 100 tasks (mean +/- std) ---")
    print_summary(data)


if __name__ == "__main__":
    main()
