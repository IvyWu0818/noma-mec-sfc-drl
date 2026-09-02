"""
eval_final_compare_v20_fixed.py
Corrected 5-way comparison: SAC (V20) vs TD3 (V17-trained) vs PPO (V17-trained)
vs Greedy vs GA -- all five evaluated under IIoTEnvV20's delay formula
(delay = t_ul + t_comp + t_link), NOT IIoTEnvV17.

Why this script exists instead of just re-running eval_final_compare_v20.py:
eval_final_compare_v20.py delegates run_episode()/ga_search() to
experiments/baselines_v17.py, which is hardcoded to instantiate
IIoTEnvV17. Commit 397fd14 later added a local-compute delay branch
(delay = max(t_local, t_offload)) to envs/iiot_env_v17.py only --
envs/iiot_env_v20.py was never touched, so it still has the original
formula that SAC was trained on (and that TD3/PPO were trained on,
since train_td3_v17.py / train_ppo_v17.py ran before that commit).
Re-running eval_final_compare_v20.py today silently re-evaluates every
agent under the now-diverged V17 formula, which SAC in particular was
never trained to handle (it learned to use partial offloading,
rho<1, which is exactly the case the new branch changes), producing
wildly degraded (but meaningless) reward/delay numbers that have
nothing to do with model quality.

This script is self-contained (mirrors the greedy_policy/run_episode/
ga_search logic already used in rerun_ga300_v20_formula.py) rather
than modifying the shared experiments/baselines_v17.py, and forces
device="cpu" for all three SB3 models so the runtime comparison in
Fig. 9 is genuinely CPU-only across all five methods (fixing the
separate device="auto" issue where model.load() defaults to CUDA on a
GPU-equipped machine).

Run: python -m experiments.eval_final_compare_v20_fixed
"""

import json
import os
import time

import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from envs.iiot_env_v20 import MAX_NOMA_PER_CH, IIoTEnvV20

ACTION_DIM = 16
NUM_TASKS = 100
N_EVAL_EPISODES = 20

GENERATIONS = 300
SIGMA_DECAY_GENERATIONS = 100
POP_SIZE = 24
ELITE_FRAC = 0.25
MUTATION_SIGMA = 0.15
PLACEMENT_RESHUFFLE_PROB = 0.08

OUTPUT_FILE = "results/final_compare_v20.json"
BACKUP_FILE = "results/final_compare_v20_v17env_bug_backup.json"

DRL_MODELS = {
    "SAC": (SAC, "models/sac_iiot_v20_final"),
    "TD3": (TD3, "models/td3_iiot_v17_final"),
    "PPO": (PPO, "models/ppo_iiot_v17_final"),
}
ALGOS = ["SAC", "TD3", "PPO", "Greedy", "GA"]

METRIC_KEYS = [
    "episode_rewards",
    "episode_avg_delay",
    "episode_avg_slack",
    "episode_timeout_ratio",
    "episode_cpu_viol_rate",
    "episode_avg_t_ul",
    "episode_avg_t_comp",
    "episode_avg_t_link",
    "episode_avg_deadline_pressure",
    "episode_channel_overflow_ratio",
    "episode_avg_rho",
]


def greedy_policy(env, obs=None):
    """Hardcoded for IIoTEnvV20's fixed 3-MEC/3-channel/16-dim action layout."""
    mec_names = ["mec0", "mec1", "mec2"]
    task = env.tasks[env.current_idx]
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    avail = {
        n: max(env.mec_nodes[n].cpu_capacity - env.mec_nodes[n].queue_load, 0.0)
        for n in mec_names
    }

    for i, vnf in enumerate(task.sfc_chain.vnfs):
        sel_node = max(mec_names, key=lambda n: avail[n])
        sel_idx = mec_names.index(sel_node)
        action[i * 3 + sel_idx] = 1.0

        node_cap = env.mec_nodes[sel_node].cpu_capacity
        f_min = max(
            vnf.cpu_cycles / max(task.deadline * 0.6, 1.0),
            node_cap * 0.12,
        )
        f_alloc = float(np.clip(avail[sel_node] / 3.0, f_min, node_cap))
        action[9 + i] = f_alloc / node_cap

        avail[sel_node] = max(avail[sel_node] - f_alloc, 0.0)

    gains = env._task_channel_gains[env.current_idx]
    candidates = [k for k in range(3) if env._slot_ch_count[k] < MAX_NOMA_PER_CH]
    if not candidates:
        candidates = list(range(3))
    best_ch = max(candidates, key=lambda k: gains[k])
    action[12 + best_ch] = 1.0

    action[15] = 1.0
    return action


def run_episode(seed, num_tasks=100, policy_fn=None, action_seq=None, record_actions=False):
    env = IIoTEnvV20(num_tasks=num_tasks, seed=seed)
    obs, _ = env.reset(seed=seed)

    ep_reward = 0.0
    infos = []
    actions = [] if record_actions else None

    for t in range(num_tasks):
        if action_seq is not None:
            action = np.asarray(action_seq[t], dtype=np.float32)
        else:
            action = policy_fn(env, obs)

        if record_actions:
            actions.append(np.array(action, dtype=np.float32))

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        infos.append(info)
        if terminated or truncated:
            break

    if record_actions:
        return ep_reward, infos, np.array(actions, dtype=np.float32)
    return ep_reward, infos


def greedy_action_sequence(seed, num_tasks=100):
    _, _, actions = run_episode(seed, num_tasks, policy_fn=greedy_policy, record_actions=True)
    return actions


def aggregate_episode(ep_reward, infos):
    def _mean(key):
        return float(np.mean([info[key] for info in infos])) if infos else 0.0

    n = max(len(infos), 1)
    timeouts = sum(1 for info in infos if info["slack"] > 0)

    return {
        "episode_rewards": float(ep_reward),
        "episode_avg_delay": _mean("delay"),
        "episode_avg_slack": _mean("slack"),
        "episode_timeout_ratio": timeouts / n,
        "episode_cpu_viol_rate": _mean("cpu_viol_rate"),
        "episode_avg_t_ul": _mean("t_ul"),
        "episode_avg_t_comp": _mean("t_comp"),
        "episode_avg_t_link": _mean("t_link"),
        "episode_avg_deadline_pressure": _mean("deadline_pressure"),
        "episode_channel_overflow_ratio": _mean("channel_overflow"),
        "episode_avg_rho": _mean("rho"),
    }


def ga_search(seed, num_tasks=100, generations=GENERATIONS,
              sigma_decay_generations=SIGMA_DECAY_GENERATIONS):
    rng = np.random.default_rng(seed + 100_000)
    chrom_shape = (num_tasks, ACTION_DIM)
    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))

    greedy_seq = greedy_action_sequence(seed, num_tasks)

    def reshuffle_placement(ind, prob):
        out = ind.copy()
        mask = rng.random(num_tasks) < prob
        n = int(mask.sum())
        if n:
            out[mask, :9] = rng.uniform(0.0, 1.0, (n, 9))
        return out

    population = []
    for i in range(POP_SIZE):
        if i % 3 == 0:
            ind = greedy_seq + rng.normal(0.0, MUTATION_SIGMA, chrom_shape)
        elif i % 3 == 1:
            ind = reshuffle_placement(greedy_seq, 0.3)
        else:
            ind = rng.uniform(0.0, 1.0, chrom_shape)
        population.append(np.clip(ind, 0.0, 1.0).astype(np.float32))

    best_ind, best_fit = None, -np.inf

    for gen in range(generations):
        fitness = np.array([run_episode(seed, num_tasks, action_seq=ind)[0] for ind in population])

        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fit:
            best_fit = float(fitness[gen_best_idx])
            best_ind = population[gen_best_idx].copy()

        order = np.argsort(fitness)[::-1]
        elites = [population[i] for i in order[:n_elite]]

        sigma = MUTATION_SIGMA * max(0.0, 1.0 - gen / sigma_decay_generations)
        new_population = list(elites)
        while len(new_population) < POP_SIZE:
            p1, p2 = elites[rng.integers(0, n_elite)], elites[rng.integers(0, n_elite)]
            mask = rng.random(chrom_shape) < 0.5
            child = np.where(mask, p1, p2)
            child = child + rng.normal(0.0, sigma, chrom_shape)
            child = reshuffle_placement(child, PLACEMENT_RESHUFFLE_PROB)
            new_population.append(np.clip(child, 0.0, 1.0).astype(np.float32))

        population = new_population

    return best_ind


def main():
    policies = {}
    for name, (cls, path) in DRL_MODELS.items():
        model = cls.load(path, device="cpu")

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

        t1 = time.perf_counter()
        best_seq = ga_search(seed, NUM_TASKS)
        ep_reward, infos = run_episode(seed, NUM_TASKS, action_seq=best_seq)
        results["GA"]["episode_runtime_sec"].append(time.perf_counter() - t1)
        for k, v in aggregate_episode(ep_reward, infos).items():
            results["GA"][k].append(v)

        print(f"  seed {seed:2d}/{N_EVAL_EPISODES} done "
              f"(SAC delay={results['SAC']['episode_avg_delay'][-1]:.3f}, "
              f"{time.time() - t0:.1f}s elapsed)")

    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            old = json.load(f)
        with open(BACKUP_FILE, "w") as f:
            json.dump(old, f, indent=2)
        print(f"backed up previous (V17-formula-contaminated) results to {BACKUP_FILE}")

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved corrected (V20-formula, CPU-only) results to {OUTPUT_FILE}")

    print(f"\n--- Final comparison (SAC vs TD3 vs PPO vs Greedy vs GA), "
          f"all under IIoTEnvV20, CPU-only: "
          f"{N_EVAL_EPISODES} episodes x {NUM_TASKS} tasks (mean +/- std) ---")
    for name in ALGOS:
        r = results[name]
        print(f"  {name:8s} reward={np.mean(r['episode_rewards']):8.3f} "
              f"delay={np.mean(r['episode_avg_delay']):7.3f}ms "
              f"runtime={np.mean(r['episode_runtime_sec']):7.3f}s")


if __name__ == "__main__":
    main()
