"""
eval_decision_time_v17.py
Fold TD3/Greedy's per-task decision time into the end-to-end delay metric, to
check whether TD3's delay advantage over Greedy still holds once the cost of
computing the action itself is counted (not just the simulated task delay).

Where "每回合運算時間" is currently recorded: neither IIoTEnvV17.step() nor
baselines_v17.run_episode() times anything internally -- run_episode() is a
pure simulation loop. The wall-clock timing exists one layer up, in the eval
scripts: eval_baselines_v17.py (TD3/Greedy/GA/Random) and eval_drl_compare_v17.py
(TD3/DDPG/SAC/PPO) both wrap time.perf_counter() around each run_episode()
call and store it as "episode_runtime_sec". Only eval_baselines_v17.py has
both TD3 and Greedy in the same run, so this script follows that script's
timing methodology rather than eval_drl_compare_v17.py's (which never
evaluates Greedy).

decision_time_per_task_ms = episode_runtime_sec / num_tasks * 1000
delay_with_decision_time  = avg_delay (per task) + decision_time_per_task_ms

GA is intentionally excluded from the comparison table -- see the printed
note for why folding its runtime into a per-task decision cost the same way
would be an apples-to-oranges comparison.

Run: python -m experiments.eval_decision_time_v17
"""

import time

import numpy as np
from stable_baselines3 import TD3

from experiments.baselines_v17 import run_episode, greedy_policy

NUM_TASKS       = 100
N_EVAL_EPISODES = 20
MODEL_PATH      = "models/td3_iiot_v17_final"


def evaluate(policy_fn, num_tasks, n_eval_episodes):
    delays_no_dt, decision_times_ms, runtimes_sec = [], [], []
    for seed in range(n_eval_episodes):
        t0 = time.perf_counter()
        _, infos = run_episode(seed, num_tasks, policy_fn=policy_fn)
        runtime_sec = time.perf_counter() - t0

        avg_delay = float(np.mean([i["delay"] for i in infos]))
        decision_time_ms = runtime_sec / num_tasks * 1000

        delays_no_dt.append(avg_delay)
        decision_times_ms.append(decision_time_ms)
        runtimes_sec.append(runtime_sec)

    delays_no_dt = np.array(delays_no_dt)
    decision_times_ms = np.array(decision_times_ms)
    delays_with_dt = delays_no_dt + decision_times_ms

    return dict(
        delay_no_dt_mean=float(delays_no_dt.mean()), delay_no_dt_std=float(delays_no_dt.std()),
        decision_time_ms_mean=float(decision_times_ms.mean()), decision_time_ms_std=float(decision_times_ms.std()),
        delay_with_dt_mean=float(delays_with_dt.mean()), delay_with_dt_std=float(delays_with_dt.std()),
        episode_runtime_sec_mean=float(np.mean(runtimes_sec)),
    )


def main():
    model = TD3.load(MODEL_PATH)

    def td3_policy(env, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    td3 = evaluate(td3_policy, NUM_TASKS, N_EVAL_EPISODES)
    greedy = evaluate(greedy_policy, NUM_TASKS, N_EVAL_EPISODES)

    print(f"\n--- TD3 vs Greedy: delay with/without per-task decision time "
          f"({N_EVAL_EPISODES} episodes x {NUM_TASKS} tasks) ---\n")
    header = f"  {'':<10}{'delay (no DT)':>16}{'decision time':>16}{'delay (with DT)':>18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in [("TD3", td3), ("Greedy", greedy)]:
        print(f"  {name:<10}"
              f"{m['delay_no_dt_mean']:>10.4f}±{m['delay_no_dt_std']:<5.4f}"
              f"{m['decision_time_ms_mean']:>11.4f}ms "
              f"{m['delay_with_dt_mean']:>12.4f}±{m['delay_with_dt_std']:<5.4f}")

    adv_no_dt = (greedy["delay_no_dt_mean"] - td3["delay_no_dt_mean"]) / greedy["delay_no_dt_mean"] * 100
    adv_with_dt = (greedy["delay_with_dt_mean"] - td3["delay_with_dt_mean"]) / greedy["delay_with_dt_mean"] * 100

    print(f"\n  TD3 advantage vs Greedy, delay only (no decision time): {adv_no_dt:+.2f}%")
    print(f"  TD3 advantage vs Greedy, delay + decision time:          {adv_with_dt:+.2f}%")
    still_holds = "仍然成立" if adv_with_dt > 0 else "不再成立（Greedy 反而更快）"
    print(f"  => 計入決策時間後，TD3 相對 Greedy 的延遲優勢{still_holds}")

    print(f"\n  [GA 不列入本比較表]")
    print(f"  GA 是離線方法：它的『運算時間』(前一輪測得約 1.9~12.5 s/episode，"
          f"依代數而定) 是在整個 episode 開始前，一次性搜尋出全部 {NUM_TASKS} 個任務的"
          f"完整動作序列所花的時間，不是任務抵達當下即時算出的『每任務決策時間』。")
    print(f"  把它除以 {NUM_TASKS} 平攤成『每任務決策時間』會嚴重低估其真實成本"
          f"（真實系統中，任務是即時抵達的，GA 無法像 TD3/Greedy 一樣逐一即時決策，"
          f"必須等離線搜尋全部跑完才能開始執行第一個任務），因此不與 TD3/Greedy 的"
          f"『線上逐任務決策時間』放在同一張表比較。")


if __name__ == "__main__":
    main()
