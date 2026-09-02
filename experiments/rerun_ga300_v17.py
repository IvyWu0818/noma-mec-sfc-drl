"""
rerun_ga300_v17.py
Re-run only the GA baseline in results/baseline_eval_v17.json at
generations=300 (matching ga_convergence_v17.py's sigma_decay_generations=100
so it's on equal footing with the 100/200/300-gen convergence checkpoints),
collecting the full METRIC_KEYS set (not just delay/runtime). TD3/Greedy/Random
entries are left untouched since they're already v17, 20-episode, seed 0-19
results independent of GA's generation count.

Run: python -m experiments.rerun_ga300_v17
"""

import json
import time

import numpy as np

from experiments.baselines_v17 import (
    METRIC_KEYS,
    run_episode,
    ga_search,
    aggregate_episode,
)

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
GENERATIONS     = 300
SIGMA_DECAY_GENERATIONS = 100
RESULTS_FILE    = "results/baseline_eval_v17.json"
BACKUP_FILE     = "results/baseline_eval_v17_ga100_backup.json"


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    with open(BACKUP_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"backed up previous (100-gen GA) results to {BACKUP_FILE}")

    ga_results = {k: [] for k in METRIC_KEYS}
    ga_results["episode_runtime_sec"] = []

    t0 = time.time()
    for seed in range(N_EVAL_EPISODES):
        t1 = time.perf_counter()
        action_seq = ga_search(seed, NUM_TASKS, generations=GENERATIONS,
                                sigma_decay_generations=SIGMA_DECAY_GENERATIONS)
        ep_reward, infos = run_episode(seed, NUM_TASKS, action_seq=action_seq)
        ga_results["episode_runtime_sec"].append(time.perf_counter() - t1)
        for k, v in aggregate_episode(ep_reward, infos).items():
            ga_results[k].append(v)

        print(f"  seed {seed:2d}/{N_EVAL_EPISODES} done "
              f"(delay={ga_results['episode_avg_delay'][-1]:.3f}, "
              f"reward={ga_results['episode_rewards'][-1]:.3f}, "
              f"{time.time() - t0:.1f}s elapsed)")

    results["GA"] = ga_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved GA(300-gen) results into {RESULTS_FILE}")

    for k in METRIC_KEYS:
        vals = ga_results[k]
        print(f"  {k:32s} {np.mean(vals):8.4f} +/- {np.std(vals):.4f}")
    rt = ga_results["episode_runtime_sec"]
    print(f"  {'episode_runtime_sec':32s} {np.mean(rt):8.4f} +/- {np.std(rt):.4f}")


if __name__ == "__main__":
    main()
