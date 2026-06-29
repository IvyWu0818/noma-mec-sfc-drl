"""
eval_drl_compare_v17.py
Evaluate TD3 / DDPG / SAC / PPO (all trained 1M timesteps on IIoTEnvV17) with
deterministic policies, using the same per-episode seeds for a fair comparison.

Run: python -m experiments.eval_drl_compare_v17
"""

import os
import json
import time

from stable_baselines3 import TD3, DDPG, SAC, PPO

from experiments.baselines_v17 import (
    METRIC_KEYS,
    run_episode,
    aggregate_episode,
    print_summary,
)

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
OUTPUT_FILE     = "results/drl_eval_v17.json"

MODELS = {
    "TD3":  (TD3,  "models/td3_iiot_v17_final"),
    "DDPG": (DDPG, "models/ddpg_iiot_v17_final"),
    "SAC":  (SAC,  "models/sac_iiot_v17_final"),
    "PPO":  (PPO,  "models/ppo_iiot_v17_final"),
}


def main():
    policies = {}
    for name, (cls, path) in MODELS.items():
        model = cls.load(path)

        def make_policy(m):
            def policy(env, obs):
                action, _ = m.predict(obs, deterministic=True)
                return action
            return policy

        policies[name] = make_policy(model)

    results = {name: {k: [] for k in METRIC_KEYS} for name in MODELS}
    for name in results:
        results[name]["episode_runtime_sec"] = []

    t0 = time.time()
    for seed in range(N_EVAL_EPISODES):
        for name, policy_fn in policies.items():
            t1 = time.perf_counter()
            ep_reward, infos = run_episode(seed, NUM_TASKS, policy_fn=policy_fn)
            results[name]["episode_runtime_sec"].append(time.perf_counter() - t1)
            for k, v in aggregate_episode(ep_reward, infos).items():
                results[name][k].append(v)

        print(f"  seed {seed:2d}/{N_EVAL_EPISODES} done "
              f"({time.time() - t0:.1f}s elapsed)")

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUTPUT_FILE}")

    print(f"\n--- DRL comparison: {N_EVAL_EPISODES} episodes x "
          f"{NUM_TASKS} tasks (mean +/- std) ---")
    print_summary(results)


if __name__ == "__main__":
    main()
