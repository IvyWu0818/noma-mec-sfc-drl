"""
ga_convergence_v17.py
GA(V17) convergence analysis: run ga_search() out to 300 generations across
N eval seeds, plot the fitness-vs-generation convergence curve, and compare
end-to-end delay / wall-clock computation time at 100 vs 200 vs 300
generations to check whether 100 generations was already enough.

Run: python -m experiments.ga_convergence_v17
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt

from experiments.baselines_v17 import ga_search, run_episode

NUM_TASKS   = 100
N_SEEDS     = 20
GENERATIONS = 300
CHECKPOINTS = (100, 200, 300)
SIGMA_DECAY_GENERATIONS = 100   # anneal sigma->0 by gen 100 regardless of GENERATIONS,
                                # so 100/200/300-gen checkpoints are on equal footing

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
            sigma_decay_generations=SIGMA_DECAY_GENERATIONS,
        )
        fitness_curves.append(history["fitness_per_gen"])

        for g in CHECKPOINTS:
            ck = history["checkpoints"][g]
            _, infos = run_episode(seed, NUM_TASKS, action_seq=ck["best_ind"])
            checkpoint_delay[g].append(float(np.mean([i["delay"] for i in infos])))
            checkpoint_elapsed[g].append(ck["elapsed_sec"])

        print(f"  seed {seed:2d}/{N_SEEDS} done "
              f"(gen{CHECKPOINTS[0]} delay={checkpoint_delay[CHECKPOINTS[0]][-1]:.3f}, "
              f"gen{CHECKPOINTS[-1]} delay={checkpoint_delay[CHECKPOINTS[-1]][-1]:.3f})")

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

    c0, c1, c2 = CHECKPOINTS
    d0, d1, d2 = summary[c0]["delay_mean"], summary[c1]["delay_mean"], summary[c2]["delay_mean"]
    gain_0_to_1 = (d0 - d1) / d0 * 100
    gain_1_to_2 = (d1 - d2) / d1 * 100 if d1 != 0 else 0.0
    gain_0_to_2 = (d0 - d2) / d0 * 100
    print(f"\n  delay improvement {c0}->{c1} gens : {gain_0_to_1:+.2f}%")
    print(f"  delay improvement {c1}->{c2} gens: {gain_1_to_2:+.2f}%")
    print(f"  delay improvement {c0}->{c2} gens: {gain_0_to_2:+.2f}%")
    verdict = (f"{c0} 代已大致收斂（{c0}->{c2} 代延遲改善 <2%）"
               if abs(gain_0_to_2) < 2.0 else
               f"{c0} 代尚未收斂，延長代數仍有明顯延遲改善")
    print(f"  => {verdict}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "fitness_mean_per_gen": mean_curve.tolist(),
            "fitness_std_per_gen":  std_curve.tolist(),
            "checkpoint_summary":   summary,
            "delay_improvement_pct": {
                f"{c0}_to_{c1}": gain_0_to_1,
                f"{c1}_to_{c2}": gain_1_to_2,
                f"{c0}_to_{c2}": gain_0_to_2,
            },
        }, f, indent=2)
    print(f"saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
