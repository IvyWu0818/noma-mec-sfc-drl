"""
plot_scale_sweep_system_size_decomp_v17.py
TD3(V17) delay decomposition (upload / compute / link) vs. system size
(3/5/7 MEC x 3/5 CH, 100 tasks), from the last 10% of training episodes in
each config's results/td3_v17_tasks100_mec{M}_ch{C}_training_metrics.json
(produced by scale_sweep_v17.py's per-config training run).

Uses the trailing 10% of training episodes (not a fresh deterministic eval)
as a proxy for converged-policy behavior per config, since only the
3MEC/3CH baseline has a saved eval checkpoint (td3_iiot_v17_final); the other
three configs' trained models were not retained locally, only their training
metrics.

Run: python -m experiments.plot_scale_sweep_system_size_decomp_v17
"""

import json

import numpy as np
import matplotlib.pyplot as plt

CONFIGS = [
    ("3 MEC/3 CH", "results/td3_v17_tasks100_mec3_ch3_training_metrics.json"),
    ("5 MEC/3 CH", "results/td3_v17_tasks100_mec5_ch3_training_metrics.json"),
    ("7 MEC/3 CH", "results/td3_v17_tasks100_mec7_ch3_training_metrics.json"),
    ("3 MEC/5 CH", "results/td3_v17_tasks100_mec3_ch5_training_metrics.json"),
]
TAIL_FRAC = 0.10

OUTPUT_PNG = "results/figures_baseline/scale_sweep_v17_system_size_decomp.png"

COLORS = {"Upload": "#7dc3e8", "Computation": "#f0a070", "Link": "#82c882"}


def savefig_both(fig, png_path, **kwargs):
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(png_path.replace(".png", ".svg"), **kwargs)


def tail_mean(vals, frac):
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]))


def main():
    labels, t_ul, t_comp, t_link = [], [], [], []
    for label, path in CONFIGS:
        with open(path) as f:
            d = json.load(f)
        labels.append(label)
        t_ul.append(tail_mean(d["episode_avg_t_ul"], TAIL_FRAC))
        t_comp.append(tail_mean(d["episode_avg_t_comp"], TAIL_FRAC))
        t_link.append(tail_mean(d["episode_avg_t_link"], TAIL_FRAC))

    t_ul, t_comp, t_link = map(np.array, (t_ul, t_comp, t_link))
    totals = t_ul + t_comp + t_link

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(x, t_ul, color=COLORS["Upload"], label="Upload")
    ax.bar(x, t_comp, bottom=t_ul, color=COLORS["Computation"], label="Computation")
    ax.bar(x, t_link, bottom=t_ul + t_comp, color=COLORS["Link"], label="Link")

    for xi, total in zip(x, totals):
        ax.annotate(f"{total:.2f}", xy=(xi, total), xytext=(0, 4),
                     textcoords="offset points", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Delay Decomposition vs. System Size (100 tasks)")
    ax.set_ylim(0, totals.max() * 1.15)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    savefig_both(fig, OUTPUT_PNG)
    print(f"saved {OUTPUT_PNG} (+ .svg)")
    for label, u, c, l, t in zip(labels, t_ul, t_comp, t_link, totals):
        print(f"  {label:12s} upload={u:.3f}  comp={c:.3f}  link={l:.3f}  total={t:.3f}")


if __name__ == "__main__":
    main()
