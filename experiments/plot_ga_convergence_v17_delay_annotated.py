"""
plot_ga_convergence_v17_delay_annotated.py
GA(V17) convergence figure from results/ga_convergence_v17.json, annotated
with end-to-end delay (ms) at the 100/200/300-generation checkpoints (instead
of "gen=X"), plus a reference line at generation 15 -- the GA generation
count used elsewhere as the fast/default setting (ga_search()'s own
checkpoint_generations default and scale_sweep_v17.py's --ga_generations
default) -- to visually contextualize how far from converged that setting is.

Uses colors consistent with the rest of this project's GA figures
(#7E57C2 purple), not the reference mockup's blue -- see conversation.

Run: python -m experiments.plot_ga_convergence_v17_delay_annotated
"""

import json

import numpy as np
import matplotlib.pyplot as plt

INPUT_JSON  = "results/ga_convergence_v17.json"
OUTPUT_PNG  = "results/figures_ga_convergence_v17/ga_convergence_v17_delay_annotated.png"

GA_COLOR       = "#7E57C2"
PREV_GEN_COLOR = "#E07B54"
PREV_GEN       = 15


def savefig_both(fig, png_path, **kwargs):
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(png_path.replace(".png", ".svg"), **kwargs)


def main():
    with open(INPUT_JSON) as f:
        data = json.load(f)

    mean = np.array(data["fitness_mean_per_gen"])
    std  = np.array(data["fitness_std_per_gen"])
    gens = np.arange(1, len(mean) + 1)
    checkpoints = data["checkpoint_summary"]  # {"100": {...}, "200": {...}, "300": {...}}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, mean, color=GA_COLOR, lw=2, zorder=3, label="Mean fitness (20 seeds)")
    ax.fill_between(gens, mean - std, mean + std, color=GA_COLOR, alpha=0.2,
                     linewidth=0, zorder=2, label="+/- 1 std")

    ax.axvline(PREV_GEN, color=PREV_GEN_COLOR, linestyle="--", lw=1.6, zorder=1,
               label=f"Previous default ({PREV_GEN} gen)")

    for g_str, ck in sorted(checkpoints.items(), key=lambda kv: int(kv[0])):
        g = int(g_str)
        v = mean[g - 1]
        ax.axvline(g, color="gray", linestyle=":", lw=0.9, alpha=0.6, zorder=1)
        ax.scatter([g], [v], color=GA_COLOR, s=28, zorder=4)
        ax.annotate(
            f"{ck['delay_mean']:.2f} ms",
            xy=(g, v), xytext=(0, 10), textcoords="offset points",
            fontsize=10, color=PREV_GEN_COLOR, ha="center", fontweight="bold",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (cumulative reward)")
    ax.set_title("GA(V17) Convergence (300 Generations)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    savefig_both(fig, OUTPUT_PNG)
    print(f"saved {OUTPUT_PNG} (+ .svg)")


if __name__ == "__main__":
    main()
