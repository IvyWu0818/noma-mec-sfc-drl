"""
eval_drl_multiseed_v18.py
Evaluate TD3 / DDPG / SAC across 3 training seeds:
  - orig  : V17 original unseeded run
  - seed1 : V18 --seed 1
  - seed2 : V18 --seed 2

Same deterministic 20-episode x 100-task eval (seeds 0-19) as eval_drl_compare_v17.

Run: python -m experiments.eval_drl_multiseed_v18
"""

import os
import json
import time

from stable_baselines3 import TD3, DDPG, SAC

from experiments.baselines_v17 import (
    METRIC_KEYS,
    run_episode,
    aggregate_episode,
    print_summary,
)

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
OUTPUT_FILE     = "results/drl_multiseed_eval_v18.json"

MODELS = {
    "TD3":  (TD3,  {"orig":  "models/td3_iiot_v17_final",
                    "seed1": "models/td3_iiot_v18_seed1_final",
                    "seed2": "models/td3_iiot_v18_seed2_final"}),
    "DDPG": (DDPG, {"orig":  "models/ddpg_iiot_v17_final",
                    "seed1": "models/ddpg_iiot_v18_seed1_final",
                    "seed2": "models/ddpg_iiot_v18_seed2_final"}),
    "SAC":  (SAC,  {"orig":  "models/sac_iiot_v17_final",
                    "seed1": "models/sac_iiot_v18_seed1_final",
                    "seed2": "models/sac_iiot_v18_seed2_final"}),
}
SEED_LABELS = ["orig", "seed1", "seed2"]


def main():
    # results[algo][seed_label] = {metric: [per_episode_values]}
    results = {
        algo: {s: {k: [] for k in METRIC_KEYS} for s in SEED_LABELS}
        for algo in MODELS
    }

    t0 = time.time()
    for algo, (cls, paths) in MODELS.items():
        for s_label, path in paths.items():
            model = cls.load(path)

            def make_policy(m):
                def policy(env, obs):
                    action, _ = m.predict(obs, deterministic=True)
                    return action
                return policy

            policy_fn = make_policy(model)

            for ep_seed in range(N_EVAL_EPISODES):
                ep_reward, infos = run_episode(ep_seed, NUM_TASKS, policy_fn=policy_fn)
                for k, v in aggregate_episode(ep_reward, infos).items():
                    results[algo][s_label][k].append(v)

            mean_r = sum(results[algo][s_label]["episode_rewards"]) / N_EVAL_EPISODES
            print(f"  {algo:5s} {s_label:6s}  reward={mean_r:8.4f}  "
                  f"({time.time() - t0:.0f}s elapsed)")

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {OUTPUT_FILE}")

    import numpy as np
    print(f"\n--- Multi-seed DRL comparison ({N_EVAL_EPISODES} eps x {NUM_TASKS} tasks) ---")
    print(f"  {'Algo':6s}  {'seed':6s}  {'Reward':>10s}  {'Delay':>8s}  {'Timeout':>8s}  "
          f"{'CPU viol':>10s}  {'CH overflow':>12s}")
    print("  " + "-" * 70)
    for algo in MODELS:
        seed_rewards = []
        for s in SEED_LABELS:
            r = np.mean(results[algo][s]["episode_rewards"])
            d = np.mean(results[algo][s]["episode_avg_delay"])
            t = np.mean(results[algo][s]["episode_timeout_ratio"])
            c = np.mean(results[algo][s]["episode_cpu_viol_rate"])
            o = np.mean(results[algo][s]["episode_channel_overflow_ratio"])
            seed_rewards.append(r)
            print(f"  {algo:6s}  {s:6s}  {r:10.4f}  {d:8.4f}  {t:8.4f}  {c:10.4f}  {o:12.4f}")
        print(f"  {algo:6s}  {'MEAN':6s}  {np.mean(seed_rewards):10.4f}  "
              f"std={np.std(seed_rewards):.4f}")
        print()


if __name__ == "__main__":
    main()
