"""
eval_final_compare_v20.py
Final 5-way comparison: SAC (V20, primary model) vs TD3 (V17) vs PPO (V17) vs
Greedy vs GA. Same 20-episode x 100-task deterministic eval, seeds 0-19, as
eval_baselines_v17.py / eval_drl_compare_v17.py, so results are directly
comparable with those.

Run: python -m experiments.eval_final_compare_v20
"""

import os
import json
import time

from stable_baselines3 import TD3, PPO, SAC

from experiments.baselines_v17 import (
    METRIC_KEYS,
    run_episode,
    aggregate_episode,
    print_summary,
    greedy_policy,
    ga_search,
)

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
OUTPUT_FILE     = "results/final_compare_v20.json"

DRL_MODELS = {
    "SAC": (SAC, "models/sac_iiot_v20_final"),
    "TD3": (TD3, "models/td3_iiot_v17_final"),
    "PPO": (PPO, "models/ppo_iiot_v17_final"),
}
ALGOS = ["SAC", "TD3", "PPO", "Greedy", "GA"]


def main():
    policies = {}
    for name, (cls, path) in DRL_MODELS.items():
        model = cls.load(path)

        def make_policy(m):
            def policy(env, obs):
                action, _ = m.predict(obs, deterministic=True)
                return action
            return policy

        policies[name] = make_policy(model)
    policies["Greedy"] = greedy_policy

    results = {name: {k: [] for k in METRIC_KEYS} for name in ALGOS}
    for name in results:
        results[name]["episode_runtime_sec"] = []

    t0 = time.time()
    for seed in range(N_EVAL_EPISODES):
        for name in ["SAC", "TD3", "PPO", "Greedy"]:
            t1 = time.perf_counter()
            ep_reward, infos = run_episode(seed, NUM_TASKS, policy_fn=policies[name])
            results[name]["episode_runtime_sec"].append(time.perf_counter() - t1)
            for k, v in aggregate_episode(ep_reward, infos).items():
                results[name][k].append(v)

        # GA: offline search per episode, then replay the best chromosome
        t1 = time.perf_counter()
        best_seq = ga_search(seed, NUM_TASKS)
        ep_reward, infos = run_episode(seed, NUM_TASKS, action_seq=best_seq)
        results["GA"]["episode_runtime_sec"].append(time.perf_counter() - t1)
        for k, v in aggregate_episode(ep_reward, infos).items():
            results["GA"][k].append(v)

        print(f"  seed {seed:2d}/{N_EVAL_EPISODES} done "
              f"({time.time() - t0:.1f}s elapsed)")

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUTPUT_FILE}")

    print(f"\n--- Final comparison (SAC-main vs TD3 vs PPO vs Greedy vs GA): "
          f"{N_EVAL_EPISODES} episodes x {NUM_TASKS} tasks (mean +/- std) ---")
    print_summary(results)


if __name__ == "__main__":
    main()
