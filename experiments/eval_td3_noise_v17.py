"""
eval_td3_noise_v17.py
Compare TD3(V17) evaluated with deterministic actions vs. with the same
exploration noise used at training time (NormalActionNoise(sigma=0.13),
see agents/train_td3_v17.py).

Note: SB3's TD3Policy._predict() ignores the `deterministic` flag entirely
(TD3 is a deterministic-policy algorithm) -- model.predict(obs,
deterministic=False) does NOT reproduce training-time exploration noise.
The action_noise object is only applied inside collect_rollouts() during
.learn(). To genuinely evaluate "with exploration noise", the same noise
must be added manually after predict().

Run: python -m experiments.eval_td3_noise_v17
"""

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from experiments.baselines_v17 import run_episode

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
MODEL_PATH      = "models/td3_iiot_v17_final"
NOISE_SIGMA     = 0.13   # matches train_td3_v17.py's NormalActionNoise sigma


def make_policy(model, noisy: bool, noise: NormalActionNoise, action_low, action_high):
    def policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        if noisy:
            action = np.clip(action + noise(), action_low, action_high)
        return action
    return policy


def run_setting(model, noisy: bool):
    action_low  = model.action_space.low
    action_high = model.action_space.high
    noise = NormalActionNoise(mean=np.zeros(model.action_space.shape[-1]),
                               sigma=NOISE_SIGMA * np.ones(model.action_space.shape[-1]))
    policy = make_policy(model, noisy, noise, action_low, action_high)

    ep_delays          = []   # per-episode mean delay
    ep_overflow_ratios  = []   # per-episode mean channel_overflow (fraction of tasks)
    ep_overflow_counts  = []   # per-episode count of overflow tasks (devices)

    for seed in range(N_EVAL_EPISODES):
        _, infos = run_episode(seed, NUM_TASKS, policy_fn=policy)
        delays    = [info["delay"] for info in infos]
        overflows = [info["channel_overflow"] for info in infos]

        ep_delays.append(float(np.mean(delays)))
        ep_overflow_ratios.append(float(np.mean(overflows)))
        ep_overflow_counts.append(float(np.sum(overflows)))

    return {
        "avg_delay":            float(np.mean(ep_delays)),
        "overflow_ratio_mean":  float(np.mean(ep_overflow_ratios)),
        "overflow_ratio_std":   float(np.std(ep_overflow_ratios)),
        "overflow_episode_pct": float(np.mean(np.array(ep_overflow_ratios) > 0.0)),
        "overflow_devices_avg": float(np.mean(ep_overflow_counts)),
    }


def main():
    model = TD3.load(MODEL_PATH)

    results = {
        "deterministic": run_setting(model, noisy=False),
        "with_noise":    run_setting(model, noisy=True),
    }

    print(f"\n--- TD3(V17) deterministic vs. exploration-noise eval "
          f"({N_EVAL_EPISODES} episodes x {NUM_TASKS} tasks, noise sigma={NOISE_SIGMA}) ---\n")
    header = f"  {'Metric':<32}{'Deterministic':>16}{'With Noise':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = [
        ("平均端到端延遲 (ms)",          "avg_delay"),
        ("通道超載率 mean",              "overflow_ratio_mean"),
        ("通道超載率 std",               "overflow_ratio_std"),
        ("超載發生的 episode 比例",       "overflow_episode_pct"),
        ("每 episode 平均超載裝置數",     "overflow_devices_avg"),
    ]
    for label, key in rows:
        print(f"  {label:<32}{results['deterministic'][key]:>16.4f}{results['with_noise'][key]:>16.4f}")

    return results


if __name__ == "__main__":
    main()
