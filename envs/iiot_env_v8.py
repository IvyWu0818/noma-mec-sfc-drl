import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode


class IIoTEnvV8(gym.Env):
    """
    V8: multi-regime IIoT environment
    目標：
    1. 保留 V7 的可學性
    2. 增加 task heterogeneity，避免固定模板策略
    3. 讓 slack/timeout 不再幾乎永遠為 0
    4. 讓 TD3 必須根據 state 做動態決策

    Obs (15 dims):
      [data_size, deadline,
       cycles_v1, cycles_v2, cycles_v3,
       mec_rem0, mec_rem1, mec_rem2,
       q0, q1, q2,
       total_c, pressure, sinr, task_type_id]

    task_type_id:
      0 = urgent
      1 = compute-heavy
      2 = bandwidth-heavy
    """

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=50.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta
        self.reward_scale = reward_scale
        self.np_random = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.array([0.0] * 9 + [0.05, 0.05, 0.05], dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(15,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _sample_sinr(self) -> float:
        interference = float(self.np_random.uniform(0.15, 0.65))
        return 0.5 / (interference + 0.01)

    def _sinr_to_rate(self, sinr: float) -> float:
        return 9.0 * np.log2(1.0 + sinr)

    def _sample_task_regime(self):
        p = float(self.np_random.random())
        if p < 0.35:
            return 0  # urgent
        elif p < 0.70:
            return 1  # compute-heavy
        else:
            return 2  # bandwidth-heavy

    def _build_task_by_type(self, task_id: int, task_type: int) -> Task:
        rng = random.Random(int(self.np_random.integers(0, 2**31)))

        if task_type == 0:  # urgent
            data_size = rng.randint(18, 32)
            deadline = rng.randint(10, 16)
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]

        elif task_type == 1:  # compute-heavy
            data_size = rng.randint(25, 40)
            deadline = rng.randint(14, 22)
            vnfs = [VNF(j, rng.randint(18, 30)) for j in range(3)]

        else:  # bandwidth-heavy
            data_size = rng.randint(40, 65)
            deadline = rng.randint(14, 22)
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]

        task = Task(task_id, data_size, deadline, SFC(vnfs))
        task.task_type_id = task_type
        return task

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        total_c = float(sum(v.cpu_cycles for v in task.sfc_chain.vnfs))
        mec_names = ["mec0", "mec1", "mec2"]

        mec_rem = [
            max(
                0.0,
                (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity,
            )
            for n in mec_names
        ]
        queue_load_abs = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        pressure = (task.data_size + total_c) / max(task.deadline, 1)

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],
            *mec_rem,
            *queue_load_abs,
            total_c,
            float(pressure),
            float(self._current_sinr),
            float(getattr(task, "task_type_id", 0)),
        ], dtype=np.float32)

    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # 1) uplink
        ru_k = self._sinr_to_rate(self._current_sinr)
        t_ul = task.data_size / max(ru_k, 1e-6)

        # 2) placement + cpu allocation + chaining
        t_comp = 0.0
        t_link = 0.0
        prev_node = None
        node_cpu_used = {n: 0.0 for n in mec_names}

        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:]

        selected_nodes = []

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            c_idx = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]
            selected_nodes.append(sel_node)

            node_cap = self.mec_nodes[sel_node].cpu_capacity

            # V8: adaptive lower bound, but slightly tighter than V7
            f_min = max(
                vnf.cpu_cycles / max(task.deadline * 0.6, 1.0),
                node_cap * 0.12
            )
            f_alloc = float(np.clip(cpu_ratios[i] * node_cap, f_min, node_cap))

            t_comp += vnf.cpu_cycles / f_alloc

            if prev_node is not None and prev_node != sel_node:
                t_link += 2.0

            node_cpu_used[sel_node] += f_alloc
            prev_node = sel_node

        delay = t_ul + t_comp + t_link
        slack = max(0.0, delay - task.deadline)

        cpu_viol = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )

        deadline_pressure = delay / max(task.deadline, 1e-6)

        # V8 reward:
        # 讓 delay 仍是主體，但在未超時時也保留 deadline pressure 訊號
        cost = (
            1.0 * delay
            + self.beta * slack
            + 5.0 * cpu_viol
            + 0.5 * t_comp
            + 1.5 * deadline_pressure
        )
        reward = -cost / self.reward_scale

        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.65 + node_cpu_used[n] * 0.35
            )

        self._current_sinr = self._sample_sinr()

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks

        obs = self._get_obs() if not terminated else np.zeros(15, dtype=np.float32)
        return obs, reward, terminated, False, {
            "delay": delay,
            "slack": slack,
            "cpu_viol": cpu_viol,
            "t_ul": t_ul,
            "t_comp": t_comp,
            "t_link": t_link,
            "deadline_pressure": deadline_pressure,
            "task_type_id": getattr(task, "task_type_id", 0),
            "selected_nodes": selected_nodes,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.mec_nodes = {
            "mec0": MECNode("mec0", 35),
            "mec1": MECNode("mec1", 45),
            "mec2": MECNode("mec2", 55),
        }

        self.tasks = []
        for i in range(self.num_tasks):
            task_type = self._sample_task_regime()
            self.tasks.append(self._build_task_by_type(i, task_type))

        self.current_idx = 0
        self._current_sinr = self._sample_sinr()
        return self._get_obs(), {}