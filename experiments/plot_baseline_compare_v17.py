"""
plot_baseline_compare_v17.py
TD3(V17) vs Greedy vs GA vs Random -- 6-panel bar comparison
(reward, delay, delay decomposition, cpu_viol_rate, channel overflow, timeout)

Run: python -m experiments.plot_baseline_compare_v17
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.baselines_v17 import print_summary

INPUT_FILE = "results/baseline_eval_v17.json"
OUTPUT_DIR = "results/figures_baseline"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_compare_v17.png")


def savefig_both(fig, png_path, **kwargs):
    """Save fig as PNG and as a sibling .svg (vector) file."""
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg", **kwargs)

ALGOS  = ["TD3", "Greedy", "GA"]
COLORS = {"TD3": "#2E86AB", "Greedy": "#E07B54", "GA": "#7E57C2"}


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


def runtime_panel(ax, data):
    means, stds = [], []
    for a in ALGOS:
        vals = np.array(data.get(a, {}).get("episode_runtime_sec", [])) * 1000.0
        means.append(float(vals.mean()) if len(vals) else 0.0)
        stds.append(float(vals.std()) if len(vals) else 0.0)
    bars = ax.bar(ALGOS, means, yerr=stds, capsize=4,
                   color=[COLORS[a] for a in ALGOS], alpha=0.85)
    ax.set_title("Per-Episode Computation Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Runtime (ms, log scale)", fontsize=9)
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, which="both")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.1f}", ha="center", va="bottom", fontsize=8)


# (key, title, ylabel, filename) for the single-metric bar panels
SINGLE_CFG = [
    ("episode_rewards",                "Cumulative Reward",       "Reward",            "baseline_compare_v17_reward.png"),
    ("episode_avg_delay",              "End-to-End Delay (ms)",   "Avg Delay (ms)",    "baseline_compare_v17_delay.png"),
    ("episode_cpu_viol_rate",          "CPU Violation Rate",       "Rate (raw / 135)",  "baseline_compare_v17_cpu_viol.png"),
    ("episode_channel_overflow_ratio", "Channel Overflow Ratio",   "Ratio",             "baseline_compare_v17_ch_overflow.png"),
    ("episode_timeout_ratio",          "Timeout Ratio",            "Ratio",             "baseline_compare_v17_timeout.png"),
]


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE} -- run experiments.eval_baselines_v17 first")
        return

    with open(INPUT_FILE) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Individual plots ────────────────────────────────────────────────
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
    savefig_both(fig, os.path.join(OUTPUT_DIR, "baseline_compare_v17_delay_decomp.png"))
    plt.close(fig)
    print("saved baseline_compare_v17_delay_decomp.png (+ .svg)")

    fig, ax = plt.subplots(figsize=(8, 5))
    runtime_panel(ax, data)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "baseline_compare_v17_runtime.png"))
    plt.close(fig)
    print("saved baseline_compare_v17_runtime.png (+ .svg)")

    # ── 2. 6-panel overview ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    # 1. Cumulative Reward
    bar_panel(fig.add_subplot(gs[0, 0]), data, "episode_rewards",
              "Cumulative Reward", "Reward")

    # 2. Avg Delay
    bar_panel(fig.add_subplot(gs[0, 1]), data, "episode_avg_delay",
              "End-to-End Delay (ms)", "Avg Delay (ms)")

    # 3. Delay Decomposition (stacked bars)
    decomp_panel(fig.add_subplot(gs[0, 2]), data)

    # 4. CPU Violation Rate
    bar_panel(fig.add_subplot(gs[1, 0]), data, "episode_cpu_viol_rate",
              "CPU Violation Rate", "Rate (raw / 135)")

    # 5. Channel Overflow Ratio
    bar_panel(fig.add_subplot(gs[1, 1]), data, "episode_channel_overflow_ratio",
              "Channel Overflow Ratio", "Ratio")

    # 6. Timeout Ratio
    bar_panel(fig.add_subplot(gs[1, 2]), data, "episode_timeout_ratio",
              "Timeout Ratio", "Ratio")

    fig.suptitle(
        "TD3(V17) vs Greedy vs GA -- Evaluation Comparison\n"
        f"({len(data['TD3']['episode_rewards'])} episodes x 100 tasks, "
        "error bars = std across episodes)",
        fontsize=13, fontweight="bold", y=1.02
    )
    savefig_both(fig, OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_FILE} (+ .svg)")

    n = len(data['TD3']['episode_rewards'])
    print(f"\n--- Baseline comparison: {n} episodes x 100 tasks (mean +/- std) ---")
    print_summary(data)


if __name__ == "__main__":
    main()
