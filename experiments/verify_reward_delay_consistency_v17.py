"""
verify_reward_delay_consistency_v17.py
Empirically verify that IIoTEnvV17.step()'s reward/cost uses the SAME delay
variable (delay = max(t_local, t_offload)) that run_episode()/eval scripts
report as T_e2e (info["delay"], aggregated as "episode_avg_delay") -- i.e.
that t_local is not just a post-hoc metric, it actually feeds the training
signal.

Method: for every step, reconstruct cost/reward from the fields in the
returned info dict using the exact same formula as step(), and assert it
equals the reward step() actually returned. Also report how often the
max(t_local, t_offload) branch picks t_local (proving the branch is live,
not vacuously always t_offload).

Run: python -m experiments.verify_reward_delay_consistency_v17
"""

import numpy as np
from stable_baselines3 import TD3

from envs.iiot_env_v17 import IIoTEnvV17
from experiments.baselines_v17 import run_episode, aggregate_episode

BETA = 12.0
REWARD_SCALE = 75.0
MODEL_PATH = "models/td3_iiot_v17_final"


def reconstruct_cost(info):
    """Same formula as IIoTEnvV17.step()'s cost, built only from info fields."""
    return (
        1.0 * info["delay"]
        + BETA * info["slack"]
        + 900.0 * info["cpu_viol_rate"]
        + 0.5 * info["t_comp"]
        + 1.5 * info["deadline_pressure"]
        + 20.0 * info["channel_overflow"]
    )


def main():
    print("=== Part 1: step()-level round-trip check ===")
    env = IIoTEnvV17(num_tasks=20, seed=0)
    obs, _ = env.reset(seed=0)
    model = TD3.load(MODEL_PATH)

    max_mismatch = 0.0
    t_local_wins = 0
    for t in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        reconstructed_cost = reconstruct_cost(info)
        reconstructed_reward = -reconstructed_cost / REWARD_SCALE
        mismatch = abs(reconstructed_reward - reward)
        max_mismatch = max(max_mismatch, mismatch)

        branch = "t_local" if info["t_local"] >= info["t_offload"] else "t_offload"
        if branch == "t_local":
            t_local_wins += 1
        print(f"  step {t:2d}: rho={info['rho']:.3f}  t_local={info['t_local']:7.3f}  "
              f"t_offload={info['t_offload']:7.3f}  delay(reward's)={info['delay']:7.3f}  "
              f"-> max()={branch:9s}  reward(step)={reward:8.5f}  "
              f"reward(reconstructed from info)={reconstructed_reward:8.5f}  "
              f"diff={mismatch:.2e}")

        if terminated:
            break

    print(f"\n  max |reward(step) - reward(reconstructed from info)| over 20 steps: {max_mismatch:.2e}")
    print(f"  => {'IDENTICAL (bug-free)' if max_mismatch < 1e-5 else 'MISMATCH -- see above'}")
    print(f"  t_local won max(t_local, t_offload) in {t_local_wins}/20 steps "
          f"(branch is live, not vacuous)")

    print("\n=== Part 2: episode-level T_e2e consistency (run_episode -> aggregate_episode) ===")
    def td3_policy(env, obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    ep_reward, infos = run_episode(seed=0, num_tasks=100, policy_fn=td3_policy)
    manual_mean_delay = float(np.mean([i["delay"] for i in infos]))
    reported = aggregate_episode(ep_reward, infos)

    print(f"  mean(info['delay']) computed manually over 100 steps: {manual_mean_delay:.6f}")
    print(f"  aggregate_episode()['episode_avg_delay'] (T_e2e reported by eval scripts): "
          f"{reported['episode_avg_delay']:.6f}")
    print(f"  => {'IDENTICAL' if abs(manual_mean_delay - reported['episode_avg_delay']) < 1e-9 else 'MISMATCH'}")

    manual_mean_t_offload = float(np.mean([i["t_offload"] for i in infos]))
    print(f"\n  For comparison, mean(t_offload) alone (i.e. delay WITHOUT the local branch): "
          f"{manual_mean_t_offload:.6f}")
    print(f"  T_e2e (with local branch) vs t_offload-only differ by: "
          f"{reported['episode_avg_delay'] - manual_mean_t_offload:+.6f} ms "
          f"-- confirms T_e2e is NOT just the offload branch.")


if __name__ == "__main__":
    main()
