"""
plot_baseline_compare_v17_paper_style.py
IEEE-paper-style 6-panel figure ("(a)...(f)" captions below each subplot,
one "Fig. N." caption line below the whole grid) combining the
baseline_compare_v17 figures except reward:

  (a) End-to-End Delay          (b) Delay Decomposition   (c) CPU Violation Rate
  (d) Channel Overflow Ratio    (e) Timeout Ratio         (f) Per-Episode Computation Time

Reuses the panel-drawing functions from plot_baseline_compare_v17.py, but
strips their bold top titles and adds "(x) caption" text below each subplot
instead, matching typical IEEE figure conventions.

Run: python -m experiments.plot_baseline_compare_v17_paper_style
"""

import os
import json

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

from experiments.plot_baseline_compare_v17 import (
    bar_panel, decomp_panel, runtime_panel, savefig_both,
)

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 標楷體 (DFKai-SB) for the (a)-(f) sub-captions specifically.
KAI_FONT = fm.FontProperties(fname="assets/fonts/DFKai-SB.ttf", size=24)

INPUT_FILE  = "results/baseline_eval_v17.json"
OUTPUT_DIR  = "results/figures_baseline"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_compare_v17_paper_style.png")

# (panel fn, kwargs, sub-caption)
PANELS = [
    (bar_panel, dict(key="episode_avg_delay", title="", ylabel="Avg Delay (ms)"),
     "(a) 端到端延遲"),
    (decomp_panel, dict(), "(b) 延遲分解比較"),
    (bar_panel, dict(key="episode_cpu_viol_rate", title="", ylabel="Rate (raw / 135)"),
     "(c) CPU超載率比較"),
    (bar_panel, dict(key="episode_channel_overflow_ratio", title="", ylabel="Ratio"),
     "(d) 通道超載率比較"),
    (bar_panel, dict(key="episode_timeout_ratio", title="", ylabel="Ratio"),
     "(e) 超時率比較"),
    (runtime_panel, dict(), "(f) 每回合運算時間比較"),
]


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Missing {INPUT_FILE} -- run experiments.eval_baselines_v17 first")
        return

    with open(INPUT_FILE) as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(15, 8.6))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.22,
                             top=0.99, bottom=0.03, left=0.045, right=0.99)

    for (panel_fn, kwargs, caption), (row, col) in zip(
            PANELS, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]):
        ax = fig.add_subplot(gs[row, col])
        if panel_fn is bar_panel:
            bar_panel(ax, data, kwargs["key"], kwargs["title"], kwargs["ylabel"])
        else:
            panel_fn(ax, data)
        ax.set_title("")  # drop the bold top title
        ax.set_title(caption, y=-0.18, pad=0, fontproperties=KAI_FONT)

    savefig_both(fig, OUTPUT_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT_FILE} (+ .svg)")


if __name__ == "__main__":
    main()
