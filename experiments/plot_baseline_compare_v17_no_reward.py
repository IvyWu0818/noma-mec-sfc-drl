"""
plot_baseline_compare_v17_no_reward.py
TD3(V17) vs Greedy vs GA -- 6-panel overview combining the baseline_compare_v17
figures EXCEPT reward: Delay, Delay Decomposition, Per-Episode Computation
Time (runtime), CPU Violation Rate, Channel Overflow Ratio, Timeout Ratio.

Reuses the panel-drawing functions from plot_baseline_compare_v17.py so
styling/colors stay identical to the individual figures in
results/figures_baseline/.

Run: python -m experiments.plot_baseline_compare_v17_no_reward
"""

import os
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.plot_baseline_compare_v17 import (
    bar_panel, decomp_panel, runtime_panel, savefig_both,
)

INPUT_FILE  = "results/baseline_eval_v17.json"
OUTPUT_DIR  = "results/figures_baseline"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_compare_v17_no_reward.png")


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE} -- run experiments.eval_baselines_v17 first")
        return

    with open(INPUT_FILE) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    # 1. Avg Delay
    bar_panel(fig.add_subplot(gs[0, 0]), data, "episode_avg_delay",
              "End-to-End Delay (ms)", "Avg Delay (ms)")

    # 2. Delay Decomposition (stacked bars)
    decomp_panel(fig.add_subplot(gs[0, 1]), data)

    # 3. Per-Episode Computation Time (runtime, log scale)
    runtime_panel(fig.add_subplot(gs[0, 2]), data)

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


if __name__ == "__main__":
    main()
