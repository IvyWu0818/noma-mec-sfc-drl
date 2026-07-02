"""
plot_td3_v19_compare.py
Compare TD3 V17/V18 (policy_delay=2, default) vs TD3 V19 (policy_delay=1).
Also shows DDPG and SAC multi-seed results for context.

Run: python -m experiments.plot_td3_v19_compare
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

OUTPUT_DIR = "results/figures_v19"

# Training metric files
TRAIN_FILES = {
    "TD3 V17/V18\n(delay=2)": [
        "results/td3_v17_training_metrics.json",
        "results/td3_v18_seed1_training_metrics.json",
        "results/td3_v18_seed2_training_metrics.json",
    ],
    "TD3 V19\n(delay=1)": [
        "results/td3_v19_seed1_training_metrics.json",
        "results/td3_v19_seed2_training_metrics.json",
        "results/td3_v19_seed3_training_metrics.json",
    ],
}

# Eval data
V18_EVAL_FILE = "results/drl_multiseed_eval_v18.json"  # TD3/DDPG/SAC 3-seed eval
V19_EVAL_FILE = "results/td3_v19_eval.json"

COLORS = {
    "TD3 V17/V18\n(delay=2)": "#2E86AB",
    "TD3 V19\n(delay=1)":     "#E07B54",
    "DDPG":                    "#9B59B6",
    "SAC":                     "#7E57C2",
}
SEED_STYLES = ["-", "--", ":"]


def savefig_both(fig, png_path, **kwargs):
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg", **kwargs)


def smooth(x, window=100):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def training_curve_panel(ax):
    for group, paths in TRAIN_FILES.items():
        color = COLORS[group]
        for i, path in enumerate(paths):
            with open(path) as f:
                rewards = json.load(f)["episode_rewards"]
            sm = smooth(rewards, window=100)
            x  = np.arange(len(sm)) + 100
            ax.plot(x, sm, color=color, linestyle=SEED_STYLES[i],
                    alpha=0.8, linewidth=1.5)

    handles = [
        mlines.Line2D([], [], color=COLORS["TD3 V17/V18\n(delay=2)"],
                      linewidth=2, label="TD3 V17/V18 (policy_delay=2)"),
        mlines.Line2D([], [], color=COLORS["TD3 V19\n(delay=1)"],
                      linewidth=2, label="TD3 V19 (policy_delay=1)"),
        mlines.Line2D([], [], color="grey", linestyle="-",  linewidth=1.2, label="seed A"),
        mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.2, label="seed B"),
        mlines.Line2D([], [], color="grey", linestyle=":",  linewidth=1.2, label="seed C"),
    ]
    ax.legend(handles=handles, ncol=2, fontsize=9, loc="lower right")
    ax.set_title("Training Reward: TD3 policy_delay=2 vs 1 (100-ep moving avg)",
                  fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.4)


def eval_bar_panel(ax, v18_data, v19_data, key, title, ylabel):
    """Bar chart: TD3 delay=2, TD3 delay=1, DDPG, SAC (mean±std across seeds)."""
    groups = {
        "TD3\n(delay=2)": [np.mean(v18_data["TD3"][s][key]) for s in ["orig","seed1","seed2"]],
        "TD3\n(delay=1)": [np.mean(v19_data[s][key])        for s in ["seed1","seed2","seed3"]],
        "DDPG":            [np.mean(v18_data["DDPG"][s][key]) for s in ["orig","seed1","seed2"]],
        "SAC":             [np.mean(v18_data["SAC"][s][key])  for s in ["orig","seed1","seed2"]],
    }
    bar_colors = ["#2E86AB", "#E07B54", "#9B59B6", "#7E57C2"]
    labels, means, stds = [], [], []
    for label, vals in groups.items():
        labels.append(label)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))

    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                   color=bar_colors, alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)


def main():
    with open(V18_EVAL_FILE) as f:
        v18_data = json.load(f)
    with open(V19_EVAL_FILE) as f:
        v19_data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Training curve comparison ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    training_curve_panel(ax)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "td3_v19_training_curve.png"))
    plt.close(fig)
    print("saved td3_v19_training_curve.png (+ .svg)")

    # ── 2. Eval bar panels ─────────────────────────────────────────────────
    PANELS = [
        ("episode_rewards",                "Cumulative Reward",    "Reward"),
        ("episode_avg_delay",              "End-to-End Delay",     "Delay (ms)"),
        ("episode_timeout_ratio",          "Timeout Ratio",        "Ratio"),
        ("episode_channel_overflow_ratio", "Channel Overflow",     "Ratio"),
        ("episode_cpu_viol_rate",          "CPU Violation Rate",   "Rate"),
    ]

    # ── 3. Combined overview ───────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    training_curve_panel(fig.add_subplot(gs[0, :]))

    eval_bar_panel(fig.add_subplot(gs[1, 0]), v18_data, v19_data,
                   "episode_rewards", "Cumulative Reward (mean±std across seeds)", "Reward")
    eval_bar_panel(fig.add_subplot(gs[1, 1]), v18_data, v19_data,
                   "episode_avg_delay", "End-to-End Delay (ms)", "Avg Delay (ms)")
    eval_bar_panel(fig.add_subplot(gs[1, 2]), v18_data, v19_data,
                   "episode_timeout_ratio", "Timeout Ratio", "Ratio")
    eval_bar_panel(fig.add_subplot(gs[2, 0]), v18_data, v19_data,
                   "episode_channel_overflow_ratio", "Channel Overflow Ratio", "Ratio")
    eval_bar_panel(fig.add_subplot(gs[2, 1]), v18_data, v19_data,
                   "episode_cpu_viol_rate", "CPU Violation Rate", "Rate")
    eval_bar_panel(fig.add_subplot(gs[2, 2]), v18_data, v19_data,
                   "episode_avg_rho", "Avg Rho (offload fraction)", "Rho")

    fig.suptitle(
        "TD3 policy_delay=2 (V17/V18) vs policy_delay=1 (V19) vs DDPG vs SAC\n"
        "Eval: mean ± std across 3 seeds (20 eps × 100 tasks each)",
        fontsize=12, fontweight="bold", y=1.01
    )
    savefig_both(fig, os.path.join(OUTPUT_DIR, "td3_v19_compare.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved td3_v19_compare.png (+ .svg)")

    # ── Numeric summary ───────────────────────────────────────────────────
    print("\n--- Eval summary (mean ± std across 3 seeds) ---")
    header_groups = {
        "TD3 delay=2": [np.mean(v18_data["TD3"][s]["episode_rewards"]) for s in ["orig","seed1","seed2"]],
        "TD3 delay=1": [np.mean(v19_data[s]["episode_rewards"])        for s in ["seed1","seed2","seed3"]],
        "DDPG":        [np.mean(v18_data["DDPG"][s]["episode_rewards"]) for s in ["orig","seed1","seed2"]],
        "SAC":         [np.mean(v18_data["SAC"][s]["episode_rewards"])  for s in ["orig","seed1","seed2"]],
    }
    for name, vals in header_groups.items():
        print(f"  {name:14s}  reward = {np.mean(vals):8.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
