"""
baselines_v17.py
Non-DRL baseline policies for IIoTEnvV17: Greedy, Random, and an offline GA
that searches the full per-episode action sequence.
"""

import numpy as np

from envs.iiot_env_v17 import IIoTEnvV17, N_CHANNELS, MAX_NOMA_PER_CH

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

METRIC_LABELS = {
    "episode_rewards":                "Reward             ",
    "episode_avg_delay":               "Delay (ms)         ",
    "episode_avg_slack":                "Slack (ms)         ",
    "episode_timeout_ratio":            "Timeout Ratio      ",
    "episode_cpu_viol_rate":            "CPU Viol Rate      ",
    "episode_avg_t_ul":                 "  t_ul (ms)        ",
    "episode_avg_t_comp":               "  t_comp (ms)      ",
    "episode_avg_t_link":               "  t_link (ms)      ",
    "episode_avg_deadline_pressure":     "Deadline Pressure  ",
    "episode_channel_overflow_ratio":    "CH Overflow Ratio  ",
    "episode_avg_rho":                   "Avg Rho            ",
}


def greedy_policy(env: IIoTEnvV17, obs=None) -> np.ndarray:
    """Sequential per-VNF placement: each VNF independently picks whichever MEC
    currently has the most tentatively-available capacity (after accounting for
    this task's earlier VNFs), and is allocated a 1/3 fair-share of that node's
    available capacity (floored by the env's f_min). This lets backhaul (t_link)
    emerge naturally when a later VNF finds another node has more headroom.
    Channel: overflow-avoiding best-gain pick. rho: full offload."""
    mec_names = ["mec0", "mec1", "mec2"]
    task = env.tasks[env.current_idx]
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    # Tentative per-node available capacity, updated as VNFs are placed
    avail = {
        n: max(env.mec_nodes[n].cpu_capacity - env.mec_nodes[n].queue_load, 0.0)
        for n in mec_names
    }

    # 1+2. Placement + CPU ratio, decided per VNF in chain order
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

    # 3. Channel: best gain among channels that won't overflow this slot
    gains = env._task_channel_gains[env.current_idx]
    candidates = [k for k in range(N_CHANNELS) if env._slot_ch_count[k] < MAX_NOMA_PER_CH]
    if not candidates:
        candidates = list(range(N_CHANNELS))
    best_ch = max(candidates, key=lambda k: gains[k])
    action[12 + best_ch] = 1.0

    # 4. rho: full offload
    action[15] = 1.0

    return action


def random_policy(env: IIoTEnvV17, obs=None) -> np.ndarray:
    return env.action_space.sample()


def run_episode(seed, num_tasks=100, policy_fn=None, action_seq=None, record_actions=False):
    """Run one episode under a step-wise policy_fn(env, obs) or a fixed
    action_seq of shape (num_tasks, ACTION_DIM). Returns (ep_reward, infos[, actions])."""
    env = IIoTEnvV17(num_tasks=num_tasks, seed=seed)
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


def greedy_action_sequence(seed, num_tasks=100) -> np.ndarray:
    _, _, actions = run_episode(seed, num_tasks, policy_fn=greedy_policy, record_actions=True)
    return actions


def aggregate_episode(ep_reward, infos) -> dict:
    """Per-episode summary metrics, matching V17MetricsCallback's naming."""
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


def print_summary(results: dict):
    """results: {algo_name: {metric_key: [per_episode_values]}}"""
    algos = list(results.keys())
    header = f"  {'Metric':<22}" + "".join(f"  {a:>18}" for a in algos)
    print(header)
    print("  " + "-" * (24 + 20 * len(algos)))
    for key in METRIC_KEYS:
        label = METRIC_LABELS.get(key, key)
        row = f"  {label}"
        for algo in algos:
            vals = results.get(algo, {}).get(key, [])
            if vals:
                row += f"  {np.mean(vals):>8.4f} +/- {np.std(vals):>5.4f}"
            else:
                row += f"  {'N/A':>18}"
        print(row)


def ga_search(seed, num_tasks=100, pop_size=24, generations=15,
               elite_frac=0.25, mutation_sigma=0.15,
               placement_reshuffle_prob=0.08) -> np.ndarray:
    """Offline GA over the full episode's action sequence (shape (num_tasks, 16)).
    Fitness = cumulative episode reward when replayed against env(seed).

    Besides Gaussian mutation on all genes, a per-task placement-reshuffle
    mutation (re-randomizing action[:9] for that task) is applied with
    probability `placement_reshuffle_prob`. The initial population also
    includes "greedy with partially reshuffled placement" individuals. Both
    let the search explore MEC-assignment (and thus backhaul/t_link) patterns
    beyond the greedy seed's fixed placement cycle."""
    rng = np.random.default_rng(seed + 100_000)
    chrom_shape = (num_tasks, ACTION_DIM)
    n_elite = max(2, int(pop_size * elite_frac))

    greedy_seq = greedy_action_sequence(seed, num_tasks)

    def reshuffle_placement(ind, prob):
        out = ind.copy()
        mask = rng.random(num_tasks) < prob
        n = int(mask.sum())
        if n:
            out[mask, :9] = rng.uniform(0.0, 1.0, (n, 9))
        return out

    population = []
    for i in range(pop_size):
        if i % 3 == 0:
            ind = greedy_seq + rng.normal(0.0, mutation_sigma, chrom_shape)
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

        sigma = mutation_sigma * (1.0 - gen / generations)
        new_population = list(elites)
        while len(new_population) < pop_size:
            p1, p2 = elites[rng.integers(0, n_elite)], elites[rng.integers(0, n_elite)]
            mask = rng.random(chrom_shape) < 0.5
            child = np.where(mask, p1, p2)
            child = child + rng.normal(0.0, sigma, chrom_shape)
            child = reshuffle_placement(child, placement_reshuffle_prob)
            new_population.append(np.clip(child, 0.0, 1.0).astype(np.float32))

        population = new_population

    return best_ind
