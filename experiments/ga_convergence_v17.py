"""
ga_convergence_v17.py
GA(V17) convergence analysis: run ga_search() out to 100 generations (instead
of the old default of 15) across N eval seeds, plot the fitness-vs-generation
convergence curve, and compare end-to-end delay / wall-clock computation time
at 15 vs 50 vs 100 generations to check whether 15 generations was already
enough.

Run: python -m experiments.ga_convergence_v17
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from experiments.baselines_v17 import ga_search, run_episode

NUM_TASKS   = 100
N_SEEDS     = 20
GENERATIONS = 100
CHECKPOINTS = (15, 50, 100)

OUTPUT_DIR    = "results/figures_ga_convergence_v17"
OUTPUT_FIG    = os.path.join(OUTPUT_DIR, "ga_convergence_v17.png")
OUTPUT_JSON   = "results/ga_convergence_v17.json"


def savefig_both(fig, png_path, **kwargs):
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(os.path.splitext(png_path)[0] + ".svg", **kwargs)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fitness_curves = []          # (N_SEEDS, GENERATIONS)
    checkpoint_delay = {g: [] for g in CHECKPOINTS}
    checkpoint_elapsed = {g: [] for g in CHECKPOINTS}

    for seed in range(N_SEEDS):
        best_ind, history = ga_search(
            seed, NUM_TASKS, generations=GENERATIONS,
            checkpoint_generations=CHECKPOINTS, return_history=True,
        )
        fitness_curves.append(history["fitness_per_gen"])

        for g in CHECKPOINTS:
            ck = history["checkpoints"][g]
            _, infos = run_episode(seed, NUM_TASKS, action_seq=ck["best_ind"])
            checkpoint_delay[g].append(float(np.mean([i["delay"] for i in infos])))
            checkpoint_elapsed[g].append(ck["elapsed_sec"])

        print(f"  seed {seed:2d}/{N_SEEDS} done "
              f"(gen15 delay={checkpoint_delay[15][-1]:.3f}, "
              f"gen100 delay={checkpoint_delay[100][-1]:.3f})")

    fitness_curves = np.array(fitness_curves)          # (N_SEEDS, GENERATIONS)
    mean_curve = fitness_curves.mean(axis=0)
    std_curve  = fitness_curves.std(axis=0)
    gens = np.arange(1, GENERATIONS + 1)

    # ── 收斂曲線圖 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, mean_curve, color="#7E57C2", lw=2, label="GA best fitness (mean over 20 seeds)")
    ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                     color="#7E57C2", alpha=0.2, label="+/- 1 std")
    for g in CHECKPOINTS:
        ax.axvline(g, color="gray", linestyle="--", alpha=0.6)
        ax.text(g, mean_curve[g - 1], f" gen={g}", fontsize=9, va="bottom")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best-so-far fitness (episode reward)")
    ax.set_title("GA(V17) Convergence: Fitness vs. Generation (100 tasks x 20 seeds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    savefig_both(fig, OUTPUT_FIG)
    print(f"\nsaved {OUTPUT_FIG} (+ .svg)")

    # ── 15 / 50 / 100 代比較表 ──────────────────────────────────
    print(f"\n--- GA(V17) generations comparison ({N_SEEDS} seeds x {NUM_TASKS} tasks) ---")
    print(f"  {'Generations':<14}{'Delay (ms)':>18}{'Runtime (s/episode)':>22}")
    summary = {}
    for g in CHECKPOINTS:
        d_mean, d_std = np.mean(checkpoint_delay[g]), np.std(checkpoint_delay[g])
        t_mean, t_std = np.mean(checkpoint_elapsed[g]), np.std(checkpoint_elapsed[g])
        summary[g] = dict(delay_mean=float(d_mean), delay_std=float(d_std),
                           runtime_mean=float(t_mean), runtime_std=float(t_std))
        print(f"  {g:<14}{d_mean:>10.4f} +/- {d_std:<5.4f}{t_mean:>16.4f} +/- {t_std:<5.4f}")

    d15, d50, d100 = summary[15]["delay_mean"], summary[50]["delay_mean"], summary[100]["delay_mean"]
    gain_15_to_50  = (d15 - d50) / d15 * 100
    gain_50_to_100 = (d50 - d100) / d50 * 100 if d50 != 0 else 0.0
    gain_15_to_100 = (d15 - d100) / d15 * 100
    print(f"\n  delay improvement 15->50 gens : {gain_15_to_50:+.2f}%")
    print(f"  delay improvement 50->100 gens: {gain_50_to_100:+.2f}%")
    print(f"  delay improvement 15->100 gens: {gain_15_to_100:+.2f}%")
    verdict = ("15 代已大致收斂（15->100 代延遲改善 <2%）"
               if abs(gain_15_to_100) < 2.0 else
               "15 代尚未收斂，延長代數仍有明顯延遲改善")
    print(f"  => {verdict}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "fitness_mean_per_gen": mean_curve.tolist(),
            "fitness_std_per_gen":  std_curve.tolist(),
            "checkpoint_summary":   summary,
            "delay_improvement_pct": {
                "15_to_50": gain_15_to_50,
                "50_to_100": gain_50_to_100,
                "15_to_100": gain_15_to_100,
            },
        }, f, indent=2)
    print(f"saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
