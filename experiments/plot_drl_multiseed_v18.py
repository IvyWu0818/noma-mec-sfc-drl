"""
plot_drl_multiseed_v18.py
Multi-seed comparison: TD3 vs DDPG vs SAC, 3 training seeds each.

Panels:
  1. Training curves: 9 curves (3 algos × 3 seeds), 100-ep moving avg, grouped by algo colour
  2. Multi-seed eval bars: mean reward across seeds per algo (error bar = std across 3 seeds)
  3. Full 6-panel eval overview (reward, delay, decomp, cpu_viol, ch_overflow, timeout)
     where each bar = mean across seeds, error bar = std across seeds

Run: python -m experiments.plot_drl_multiseed_v18
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

TRAIN_FILES = {
    "TD3":  {"orig":  "results/td3_v17_training_metrics.json",
             "seed1": "results/td3_v18_seed1_training_metrics.json",
             "seed2": "results/td3_v18_seed2_training_metrics.json"},
    "DDPG": {"orig":  "results/ddpg_v17_training_metrics.json",
             "seed1": "results/ddpg_v18_seed1_training_metrics.json",
             "seed2": "results/ddpg_v18_seed2_training_metrics.json"},
    "SAC":  {"orig":  "results/sac_v17_training_metrics.json",
             "seed1": "results/sac_v18_seed1_training_metrics.json",
             "seed2": "results/sac_v18_seed2_training_metrics.json"},
}
EVAL_FILE  = "results/drl_multiseed_eval_v18.json"
OUTPUT_DIR = "results/figures_drl_multiseed"

ALGOS  = ["TD3", "DDPG", "SAC"]
COLORS = {"TD3": "#2E86AB", "DDPG": "#E07B54", "SAC": "#7E57C2"}
SEED_LABELS  = ["orig", "seed1", "seed2"]
SEED_STYLES  = {"orig": "-", "seed1": "--", "seed2": ":"}
SEED_ALPHAS  = {"orig": 0.9,  "seed1": 0.7,  "seed2": 0.7}


def savefig_both(fig, png_path, **kwargs):
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg", **kwargs)


def smooth(x, window=100):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ── helpers ────────────────────────────────────────────────────────────────

def seed_means(data, algo, key):
    """Return list of per-seed means (one per seed label)."""
    return [float(np.mean(data[algo][s][key])) for s in SEED_LABELS]


def across_seed_mean_std(data, algo, key):
    vals = seed_means(data, algo, key)
    return float(np.mean(vals)), float(np.std(vals))


# ── panel drawing ──────────────────────────────────────────────────────────

def training_curve_panel(ax):
    for algo in ALGOS:
        for s in SEED_LABELS:
            with open(TRAIN_FILES[algo][s]) as f:
                rewards = json.load(f)["episode_rewards"]
            sm = smooth(rewards, window=100)
            x  = np.arange(len(sm)) + 100
            ax.plot(x, sm, color=COLORS[algo],
                    linestyle=SEED_STYLES[s], alpha=SEED_ALPHAS[s], linewidth=1.4)

    # Legend: algo colour patches + seed-style line patches
    algo_handles = [mlines.Line2D([], [], color=COLORS[a], linewidth=2, label=a)
                    for a in ALGOS]
    seed_handles = [mlines.Line2D([], [], color="grey",
                                   linestyle=SEED_STYLES[s], linewidth=1.5,
                                   label={"orig": "orig (V17)", "seed1": "seed 1",
                                           "seed2": "seed 2"}[s])
                    for s in SEED_LABELS]
    ax.legend(handles=algo_handles + seed_handles, ncol=3, fontsize=8,
              loc="lower right")
    ax.set_title("Training Reward (100-ep moving avg, 3 seeds each)", fontsize=12,
                  fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.4)


def bar_panel(ax, data, key, title, ylabel):
    means = [across_seed_mean_std(data, a, key)[0] for a in ALGOS]
    stds  = [across_seed_mean_std(data, a, key)[1] for a in ALGOS]
    bars = ax.bar(ALGOS, means, yerr=stds, capsize=5,
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
        means = np.array([across_seed_mean_std(data, a, key)[0] for a in ALGOS])
        ax.bar(ALGOS, means, bottom=bottoms, color=color, alpha=0.85, label=label)
        bottoms += means
    ax.set_title("Delay Decomposition (mean across seeds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Delay (ms)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def main():
    for algo, seeds in TRAIN_FILES.items():
        for s, path in seeds.items():
            if not os.path.exists(path):
                print(f"Missing {path}"); return
    if not os.path.exists(EVAL_FILE):
        print(f"Missing {EVAL_FILE} -- run experiments.eval_drl_multiseed_v18 first")
        return

    with open(EVAL_FILE) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Training curves ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    training_curve_panel(ax)
    fig.tight_layout()
    savefig_both(fig, os.path.join(OUTPUT_DIR, "drl_multiseed_training_curve.png"))
    plt.close(fig)
    print("saved drl_multiseed_training_curve.png (+ .svg)")

    # ── 2. 6-panel eval overview (bars = mean across seeds, err = std across seeds) ─
    SINGLE_CFG = [
        ("episode_rewards",                "Cumulative Reward (mean ± std across seeds)", "Reward",           "drl_multiseed_reward.png"),
        ("episode_avg_delay",              "End-to-End Delay (ms)",                       "Avg Delay (ms)",   "drl_multiseed_delay.png"),
        ("episode_cpu_viol_rate",          "CPU Violation Rate",                          "Rate (raw / 135)", "drl_multiseed_cpu_viol.png"),
        ("episode_channel_overflow_ratio", "Channel Overflow Ratio",                      "Ratio",            "drl_multiseed_ch_overflow.png"),
        ("episode_timeout_ratio",          "Timeout Ratio",                               "Ratio",            "drl_multiseed_timeout.png"),
    ]
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
    savefig_both(fig, os.path.join(OUTPUT_DIR, "drl_multiseed_delay_decomp.png"))
    plt.close(fig)
    print("saved drl_multiseed_delay_decomp.png (+ .svg)")

    # ── 3. Combined 7-panel overview ──────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

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
        "TD3 vs DDPG vs SAC -- Multi-Seed Robustness (3 seeds each)\n"
        "Training curves: 100-ep moving avg  |  "
        f"Eval bars: mean ± std across 3 seeds ({N_EVAL_EPISODES} eps x 100 tasks each)",
        fontsize=12, fontweight="bold", y=1.01
    )
    savefig_both(fig,
                 os.path.join(OUTPUT_DIR, "drl_multiseed_v18.png"),
                 bbox_inches="tight")
    plt.close(fig)
    print(f"saved drl_multiseed_v18.png (+ .svg)")

    # ── Numeric summary ───────────────────────────────────────────────────
    print(f"\n--- Mean ± std across 3 seeds ({N_EVAL_EPISODES} eval eps each) ---")
    print(f"  {'Metric':<32}" + "".join(f"  {a:>22}" for a in ALGOS))
    print("  " + "-" * (34 + 24 * len(ALGOS)))
    KEYS = [
        ("episode_rewards",                "Reward          "),
        ("episode_avg_delay",              "Delay (ms)      "),
        ("episode_timeout_ratio",          "Timeout Ratio   "),
        ("episode_cpu_viol_rate",          "CPU Viol Rate   "),
        ("episode_channel_overflow_ratio", "CH Overflow     "),
        ("episode_avg_t_comp",             "t_comp (ms)     "),
        ("episode_avg_t_link",             "t_link (ms)     "),
        ("episode_avg_rho",                "Avg Rho         "),
    ]
    for key, label in KEYS:
        row = f"  {label}"
        for algo in ALGOS:
            m, s = across_seed_mean_std(data, algo, key)
            row += f"  {m:>9.4f} +/- {s:>6.4f}"
        print(row)


N_EVAL_EPISODES = 20   # matches eval script

if __name__ == "__main__":
    main()
