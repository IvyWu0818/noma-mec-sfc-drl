"""
scale_sweep_v17.py
Batch scalability sweep for IIoTEnvV17: TD3 (freshly trained per config) vs
Greedy vs GA, across three independent one-factor-at-a-time axes (other
settings held at the baseline: num_tasks=100, n_mec=3, n_channels=3):

  - num_tasks  : 50 / 100 / 200 / 500
  - n_mec      : 3 / 5 / 7            (n_channels held at 3)
  - n_channels : 3 / 5                (n_mec held at 3)

TD3 needs a freshly-trained model per config: changing n_mec/n_channels
changes the env's action/observation dims, so no policy is transferable
across configs. Greedy/GA need no training and are evaluated directly.

This is expensive (each config trains TD3 for --total_timesteps, default
1,000,000) -- intended to run on a GPU machine.

Fast wiring smoke-test (does NOT produce meaningful results, just checks the
whole pipeline runs end-to-end):
  python -m experiments.scale_sweep_v17 --total_timesteps 500 \
      --n_eval_episodes 2 --ga_generations 2 --ga_pop_size 6 --device cpu

Full run (GPU machine):
  python -m experiments.scale_sweep_v17
"""

import argparse
import csv
import os
import time

import numpy as np
from stable_baselines3 import TD3

from agents.train_td3_v17 import train as train_td3
from experiments.baselines_v17 import run_episode, greedy_policy, ga_search

TASK_SWEEP    = [50, 100, 200, 500]
MEC_SWEEP     = [3, 5, 7]
CHANNEL_SWEEP = [3, 5]

CSV_FIELDS = ["axis", "value", "num_tasks", "n_mec", "n_channels", "algo",
              "avg_delay", "timeout_ratio", "channel_overflow_ratio",
              "avg_episode_runtime_sec"]


def build_configs():
    """Three one-factor-at-a-time sweeps around the baseline
    (num_tasks=100, n_mec=3, n_channels=3); baseline itself only appears once."""
    configs = [dict(axis="num_tasks", value=nt, num_tasks=nt, n_mec=3, n_channels=3)
               for nt in TASK_SWEEP]
    configs += [dict(axis="n_mec", value=nm, num_tasks=100, n_mec=nm, n_channels=3)
                for nm in MEC_SWEEP if nm != 3]
    configs += [dict(axis="n_channels", value=nc, num_tasks=100, n_mec=3, n_channels=nc)
                for nc in CHANNEL_SWEEP if nc != 3]
    return configs


def model_path_for(num_tasks, n_mec, n_channels):
    if num_tasks == 100 and n_mec == 3 and n_channels == 3:
        return "models/td3_iiot_v17_final"   # canonical baseline model (already trained)
    return f"models/td3_iiot_v17_tasks{num_tasks}_mec{n_mec}_ch{n_channels}_final"


def evaluate_algo(policy_fn, num_tasks, env_kwargs, n_eval_episodes,
                   is_ga=False, ga_kwargs=None):
    delays, timeouts, overflows, runtimes = [], [], [], []
    for seed in range(n_eval_episodes):
        t0 = time.perf_counter()
        if is_ga:
            best_seq = ga_search(seed, num_tasks, env_kwargs=env_kwargs, **(ga_kwargs or {}))
            _, infos = run_episode(seed, num_tasks, action_seq=best_seq, env_kwargs=env_kwargs)
        else:
            _, infos = run_episode(seed, num_tasks, policy_fn=policy_fn, env_kwargs=env_kwargs)
        runtimes.append(time.perf_counter() - t0)
        delays.append(float(np.mean([i["delay"] for i in infos])))
        timeouts.append(float(np.mean([i["slack"] > 0 for i in infos])))
        overflows.append(float(np.mean([i["channel_overflow"] for i in infos])))
    return dict(
        avg_delay=float(np.mean(delays)),
        timeout_ratio=float(np.mean(timeouts)),
        channel_overflow_ratio=float(np.mean(overflows)),
        avg_episode_runtime_sec=float(np.mean(runtimes)),
    )


def run_config(cfg, args):
    num_tasks, n_mec, n_channels = cfg["num_tasks"], cfg["n_mec"], cfg["n_channels"]
    env_kwargs = dict(n_mec=n_mec, n_channels=n_channels)
    print(f"\n=== config: {cfg['axis']}={cfg['value']}  "
          f"(num_tasks={num_tasks}, n_mec={n_mec}, n_channels={n_channels}) ===")

    model_path = model_path_for(num_tasks, n_mec, n_channels)
    if args.skip_training and os.path.exists(model_path + ".zip"):
        print(f"  [skip_training] reusing {model_path}")
    else:
        cfg_suffix = f"_tasks{num_tasks}_mec{n_mec}_ch{n_channels}"
        train_td3(
            num_tasks=num_tasks, n_mec=n_mec, n_channels=n_channels,
            total_timesteps=args.total_timesteps, device=args.device,
            model_path=model_path,
            metrics_path=f"results/td3_v17{cfg_suffix}_training_metrics.json",
            log_prefix=f"logs/v17{cfg_suffix}_",
        )

    model = TD3.load(model_path)

    def td3_policy(env, obs, _m=model):
        action, _ = _m.predict(obs, deterministic=True)
        return action

    td3_metrics = evaluate_algo(td3_policy, num_tasks, env_kwargs, args.n_eval_episodes)
    greedy_metrics = evaluate_algo(greedy_policy, num_tasks, env_kwargs, args.n_eval_episodes)
    ga_metrics = evaluate_algo(
        None, num_tasks, env_kwargs, args.n_eval_episodes, is_ga=True,
        ga_kwargs=dict(pop_size=args.ga_pop_size, generations=args.ga_generations),
    )

    adv_greedy = (greedy_metrics["avg_delay"] - td3_metrics["avg_delay"]) / greedy_metrics["avg_delay"] * 100
    adv_ga     = (ga_metrics["avg_delay"] - td3_metrics["avg_delay"]) / ga_metrics["avg_delay"] * 100
    print(f"  TD3 delay={td3_metrics['avg_delay']:.3f}  Greedy={greedy_metrics['avg_delay']:.3f}  "
          f"GA={ga_metrics['avg_delay']:.3f}")
    print(f"  TD3 advantage vs Greedy: {adv_greedy:+.1f}%   vs GA: {adv_ga:+.1f}%")

    rows = [
        dict(axis=cfg["axis"], value=cfg["value"], num_tasks=num_tasks,
             n_mec=n_mec, n_channels=n_channels, algo=algo, **metrics)
        for algo, metrics in [("TD3", td3_metrics), ("Greedy", greedy_metrics), ("GA", ga_metrics)]
    ]
    advantage_row = dict(axis=cfg["axis"], value=cfg["value"], num_tasks=num_tasks,
                          n_mec=n_mec, n_channels=n_channels,
                          td3_vs_greedy_pct=adv_greedy, td3_vs_ga_pct=adv_ga)
    return rows, advantage_row


def print_advantage_trend(advantage_rows):
    print("\n--- TD3 relative delay advantage vs Greedy/GA across scale ---")
    for axis in ["num_tasks", "n_mec", "n_channels"]:
        axis_rows = sorted((r for r in advantage_rows if r["axis"] == axis), key=lambda r: r["value"])
        if not axis_rows:
            continue
        print(f"\n  Axis: {axis}")
        print(f"    {'value':>8}{'vs Greedy %':>14}{'vs GA %':>12}")
        for r in axis_rows:
            print(f"    {r['value']:>8}{r['td3_vs_greedy_pct']:>13.1f}%{r['td3_vs_ga_pct']:>11.1f}%")
        greedy_trend = [r["td3_vs_greedy_pct"] for r in axis_rows]
        ga_trend = [r["td3_vs_ga_pct"] for r in axis_rows]
        g_dir = "increases" if greedy_trend[-1] > greedy_trend[0] else "decreases"
        a_dir = "increases" if ga_trend[-1] > ga_trend[0] else "decreases"
        print(f"    -> advantage vs Greedy {g_dir} from {greedy_trend[0]:+.1f}% to {greedy_trend[-1]:+.1f}% "
              f"as {axis} grows")
        print(f"    -> advantage vs GA     {a_dir} from {ga_trend[0]:+.1f}% to {ga_trend[-1]:+.1f}% "
              f"as {axis} grows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--ga_pop_size", type=int, default=24)
    parser.add_argument("--ga_generations", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_csv", type=str, default="results/scale_sweep_v17.csv")
    parser.add_argument("--advantage_csv", type=str, default="results/scale_sweep_v17_advantage.csv")
    parser.add_argument("--skip_training", action="store_true",
                         help="Reuse an already-trained model at the expected path "
                              "instead of retraining (see model_path_for()).")
    args = parser.parse_args()

    all_rows, advantage_rows = [], []
    for cfg in build_configs():
        rows, adv_row = run_config(cfg, args)
        all_rows += rows
        advantage_rows.append(adv_row)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nsaved {args.output_csv}")

    adv_fields = ["axis", "value", "num_tasks", "n_mec", "n_channels",
                  "td3_vs_greedy_pct", "td3_vs_ga_pct"]
    with open(args.advantage_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=adv_fields)
        writer.writeheader()
        writer.writerows(advantage_rows)
    print(f"saved {args.advantage_csv}")

    print("\n--- Full results ---")
    header = f"  {'axis':<12}{'value':>6}{'algo':>8}{'delay':>10}{'timeout':>10}{'overflow':>10}{'runtime(s)':>12}"
    print(header)
    for r in all_rows:
        print(f"  {r['axis']:<12}{r['value']:>6}{r['algo']:>8}{r['avg_delay']:>10.3f}"
              f"{r['timeout_ratio']:>10.3f}{r['channel_overflow_ratio']:>10.3f}"
              f"{r['avg_episode_runtime_sec']:>12.4f}")

    print_advantage_trend(advantage_rows)


if __name__ == "__main__":
    main()
