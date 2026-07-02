"""
eval_td3_v19.py
Evaluate TD3 V19 (policy_delay=1) across 3 seeds and save to results/td3_v19_eval.json.
Run: python -m experiments.eval_td3_v19
"""

import os
import json
import time

import numpy as np
from stable_baselines3 import TD3

from experiments.baselines_v17 import METRIC_KEYS, run_episode, aggregate_episode

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
OUTPUT_FILE     = "results/td3_v19_eval.json"

SEED_LABELS = ["seed1", "seed2", "seed3"]
MODEL_PATHS = {
    "seed1": "models/td3_iiot_v19_seed1_final",
    "seed2": "models/td3_iiot_v19_seed2_final",
    "seed3": "models/td3_iiot_v19_seed3_final",
}


def main():
    results = {s: {k: [] for k in METRIC_KEYS} for s in SEED_LABELS}

    t0 = time.time()
    for s_label, path in MODEL_PATHS.items():
        model = TD3.load(path)

        def make_policy(m):
            def policy(env, obs):
                action, _ = m.predict(obs, deterministic=True)
                return action
            return policy

        policy_fn = make_policy(model)
        for ep_seed in range(N_EVAL_EPISODES):
            ep_reward, infos = run_episode(ep_seed, NUM_TASKS, policy_fn=policy_fn)
            for k, v in aggregate_episode(ep_reward, infos).items():
                results[s_label][k].append(v)

        mean_r = np.mean(results[s_label]["episode_rewards"])
        print(f"  TD3 V19 {s_label}  reward={mean_r:.4f}  ({time.time()-t0:.0f}s)")

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUTPUT_FILE}")

    seed_means = [np.mean(results[s]["episode_rewards"]) for s in SEED_LABELS]
    print(f"\nTD3 V19 (policy_delay=1): mean={np.mean(seed_means):.4f}  "
          f"std={np.std(seed_means):.4f}")


if __name__ == "__main__":
    main()
