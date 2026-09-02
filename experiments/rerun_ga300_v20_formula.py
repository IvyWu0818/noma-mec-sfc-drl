"""
rerun_ga300_v20_formula.py
Self-contained re-run of the GA baseline at generations=300, evaluated
against IIoTEnvV20 (the SAME formula SAC/TD3/PPO were trained/evaluated
under -- delay = t_ul + t_comp + t_link, no local-compute branch) instead
of the current envs/iiot_env_v17.py (which gained a local-compute delay
branch in commit 397fd14, AFTER the original final_compare_v20.json and
AFTER SAC v20 was trained). The current results/final_compare_v20.json's
GA entry was produced against that newer v17 formula, which is now
inconsistent with the SAC/TD3/PPO/Greedy entries in the same file --
this script fixes that by giving GA the exact formula the rest of the
comparison uses, while still getting 300-generation convergence.

Deliberately does NOT touch experiments/baselines_v17.py (shared by many
other eval/plot scripts) -- greedy_policy/run_episode/ga_search are
duplicated here, hardcoded to IIoTEnvV20's fixed 3-MEC/3-channel action
layout (same simple form baselines_v17.py used before it was generalized
for scale-sweep configs).

Run: python -m experiments.rerun_ga300_v20_formula
"""

import json
import time

import numpy as np

from envs.iiot_env_v20 import IIoTEnvV20, MAX_NOMA_PER_CH

ACTION_DIM = 16

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

NUM_TASKS               = 100
N_EVAL_EPISODES         = 20
GENERATIONS             = 300
SIGMA_DECAY_GENERATIONS = 100   # matches rerun_ga300_v17.py's schedule
POP_SIZE                = 24
ELITE_FRAC              = 0.25
MUTATION_SIGMA          = 0.15
PLACEMENT_RESHUFFLE_PROB = 0.08

FINAL_COMPARE_FILE = "results/final_compare_v20.json"
BACKUP_FILE         = "results/final_compare_v20_ga_v17formula_backup.json"
RESULTS_FILE        = "results/ga300_v20formula.json"


def greedy_policy(env, obs=None):
    """Same greedy heuristic as baselines_v17.greedy_policy's original
    (pre-scale-sweep) form: hardcoded for IIoTEnvV20's fixed 3-MEC,
    3-channel, 16-dim action layout."""
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
        "episode_rewards":                float(ep_reward),
        "episode_avg_delay":               _mean("delay"),
        "episode_avg_slack":               _mean("slack"),
        "episode_timeout_ratio":           timeouts / n,
        "episode_cpu_viol_rate":           _mean("cpu_viol_rate"),
        "episode_avg_t_ul":                _mean("t_ul"),
        "episode_avg_t_comp":              _mean("t_comp"),
        "episode_avg_t_link":              _mean("t_link"),
        "episode_avg_deadline_pressure":   _mean("deadline_pressure"),
        "episode_channel_overflow_ratio":  _mean("channel_overflow"),
        "episode_avg_rho":                 _mean("rho"),
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
    ga_results = {k: [] for k in METRIC_KEYS}
    ga_results["episode_runtime_sec"] = []

    t0 = time.time()
    for seed in range(N_EVAL_EPISODES):
        t1 = time.perf_counter()
        action_seq = ga_search(seed, NUM_TASKS)
        ep_reward, infos = run_episode(seed, NUM_TASKS, action_seq=action_seq)
        ga_results["episode_runtime_sec"].append(time.perf_counter() - t1)
        for k, v in aggregate_episode(ep_reward, infos).items():
            ga_results[k].append(v)
        print(f"  seed {seed:2d}/{N_EVAL_EPISODES} done "
              f"(delay={ga_results['episode_avg_delay'][-1]:.3f}, "
              f"reward={ga_results['episode_rewards'][-1]:.3f}, "
              f"{time.time() - t0:.1f}s elapsed)")

    with open(RESULTS_FILE, "w") as f:
        json.dump(ga_results, f, indent=2)
    print(f"\nsaved raw GA(300-gen, V20-formula) results to {RESULTS_FILE}")

    # ---- splice into final_compare_v20.json, backing up the current
    # (v17-new-formula) GA entry first ----
    with open(FINAL_COMPARE_FILE) as f:
        final_compare = json.load(f)
    with open(BACKUP_FILE, "w") as f:
        json.dump(final_compare, f, indent=2)
    print(f"backed up previous (v17-new-formula) GA results to {BACKUP_FILE}")

    final_compare["GA"] = {k: ga_results[k] for k in final_compare["SAC"].keys()}
    with open(FINAL_COMPARE_FILE, "w") as f:
        json.dump(final_compare, f, indent=2)
    print(f"spliced V20-formula 300-gen GA into {FINAL_COMPARE_FILE}")

    print(f"\n  GA reward:  {np.mean(ga_results['episode_rewards']):.3f}")
    print(f"  GA delay:   {np.mean(ga_results['episode_avg_delay']):.3f} ms")
    print(f"  GA runtime: {np.mean(ga_results['episode_runtime_sec']):.3f}s")


if __name__ == "__main__":
    main()
