"""
plot_ga_convergence_v17.py
Paper-style GA(V17) convergence figure from results/ga_convergence_v17.json
(produced by experiments/ga_convergence_v17.py). Monochrome, annotates
generations 15 / 55 / 100 with their fitness values.

Run: python -m experiments.plot_ga_convergence_v17
"""

import json

import numpy as np
import matplotlib.pyplot as plt

INPUT_JSON  = "results/ga_convergence_v17.json"
OUTPUT_PNG  = "results/ga_convergence_v17_paper.png"

ANNOTATE_GENS = (15, 55, 100)
LINE_COLOR    = "#1a1a1a"
BAND_COLOR    = "#1a1a1a"

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    with open(INPUT_JSON) as f:
        data = json.load(f)

    mean = np.array(data["fitness_mean_per_gen"])
    std  = np.array(data["fitness_std_per_gen"])
    gens = np.arange(1, len(mean) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gens, mean, color=LINE_COLOR, lw=1.8, zorder=3)
    ax.fill_between(gens, mean - std, mean + std, color=BAND_COLOR, alpha=0.15,
                     linewidth=0, zorder=2, label="±1 標準差（20 seeds）")

    for g in ANNOTATE_GENS:
        v = mean[g - 1]
        ax.axvline(g, color=LINE_COLOR, linestyle=":", lw=0.9, alpha=0.6, zorder=1)
        ax.scatter([g], [v], color=LINE_COLOR, s=22, zorder=4)
        ax.annotate(
            f"第 {g} 代\nfitness = {v:.3f}",
            xy=(g, v), xytext=(8, -28 if g != 100 else -55),
            textcoords="offset points", fontsize=9, color=LINE_COLOR,
            ha="left" if g != 100 else "right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )

    ax.set_xlabel("演化代數", fontsize=11)
    ax.set_ylabel("族群最佳適應度", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(OUTPUT_PNG, dpi=300)
    print(f"saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
