"""
eval_baselines_v17.py
Evaluate TD3(V17, best model) vs Greedy / GA / Random baselines on IIoTEnvV17,
using the same per-episode seeds for a fair comparison.

Run: python -m experiments.eval_baselines_v17
"""

import os
import json
import time

from stable_baselines3 import TD3

from experiments.baselines_v17 import (
    METRIC_KEYS,
    greedy_policy,
    random_policy,
    run_episode,
    ga_search,
    aggregate_episode,
    print_summary,
)

NUM_TASKS        = 100
N_EVAL_EPISODES  = 20
MODEL_PATH       = "models/td3_iiot_v17_final"
OUTPUT_FILE      = "results/baseline_eval_v17.json"


def main():
    model = TD3.load(MODEL_PATH)

    def td3_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    results = {name: {k: [] for k in METRIC_KEYS}
               for name in ("TD3", "Greedy", "GA", "Random")}

    t0 = time.time()
    for seed in range(N_EVAL_EPISODES):
        ep_reward, infos = run_episode(seed, NUM_TASKS, policy_fn=td3_policy)
        for k, v in aggregate_episode(ep_reward, infos).items():
            results["TD3"][k].append(v)

        ep_reward, infos = run_episode(seed, NUM_TASKS, policy_fn=greedy_policy)
        for k, v in aggregate_episode(ep_reward, infos).items():
            results["Greedy"][k].append(v)

        ep_reward, infos = run_episode(seed, NUM_TASKS, policy_fn=random_policy)
        for k, v in aggregate_episode(ep_reward, infos).items():
            results["Random"][k].append(v)

        action_seq = ga_search(seed, NUM_TASKS)
        ep_reward, infos = run_episode(seed, NUM_TASKS, action_seq=action_seq)
        for k, v in aggregate_episode(ep_reward, infos).items():
            results["GA"][k].append(v)

        print(f"  seed {seed:2d}/{N_EVAL_EPISODES} done "
              f"({time.time() - t0:.1f}s elapsed)")

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUTPUT_FILE}")

    print(f"\n--- Baseline comparison: {N_EVAL_EPISODES} episodes x "
          f"{NUM_TASKS} tasks (mean +/- std) ---")
    print_summary(results)


if __name__ == "__main__":
    main()
