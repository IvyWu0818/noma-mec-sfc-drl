import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay
from core.objective import compute_slack


class IIoTEnvV3(gym.Env):
    """
    V3: per-VNF placement + per-VNF CPU allocation

    Action (12 dims):
        [v1_m0, v1_m1, v1_m2,
         v2_m0, v2_m1, v2_m2,
         v3_m0, v3_m1, v3_m2,
         cpu1_ratio, cpu2_ratio, cpu3_ratio]

    Observation (11 dims):
        [data_size, deadline, sfc_length,
         cycles_v1, cycles_v2, cycles_v3,
         q0, q1, q2,
         total_cycles,
         deadline_pressure]
    """

    metadata = {"render_modes": []}

    def __init__(self, num_tasks=10, beta=10.0, seed=42):
        super().__init__()

        self.num_tasks = num_tasks
        self.beta = beta
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # 9 placement scores + 3 cpu ratios
        self.action_space = spaces.Box(
            low=np.array([0.0] * 9 + [0.1, 0.1, 0.1], dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32
        )

        # obs = 11 dims
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(11,),
            dtype=np.float32
        )

        self.graph = None
        self.mec_nodes = None
        self.tasks = None
        self.current_idx = 0

    def _create_mec_nodes(self):
        # 不對稱 MEC，避免 collapse 到完全對稱解
        return {
            "mec0": MECNode("mec0", 40),
            "mec1": MECNode("mec1", 55),
            "mec2": MECNode("mec2", 70)
        }

    def _create_random_task(self, task_id):
        vnfs = [VNF(i, random.randint(12, 28)) for i in range(3)]
        sfc = SFC(vnfs)

        return Task(
            task_id=task_id,
            data_size=random.randint(35, 60),
            deadline=random.randint(16, 28),
            sfc_chain=sfc
        )

    def _generate_tasks(self):
        return [self._create_random_task(i) for i in range(self.num_tasks)]

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        vnfs = task.sfc_chain.vnfs
        total_cycles = sum(v.cpu_cycles for v in vnfs)

        # deadline pressure：越大代表越緊
        deadline_pressure = (task.data_size + total_cycles) / max(task.deadline, 1)

        obs = np.array([
            float(task.data_size),
            float(task.deadline),
            float(len(vnfs)),
            float(vnfs[0].cpu_cycles),
            float(vnfs[1].cpu_cycles),
            float(vnfs[2].cpu_cycles),
            float(self.mec_nodes["mec0"].queue_load),
            float(self.mec_nodes["mec1"].queue_load),
            float(self.mec_nodes["mec2"].queue_load),
            float(total_cycles),
            float(deadline_pressure),
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.graph = create_topology()
        self.mec_nodes = self._create_mec_nodes()
        self.tasks = self._generate_tasks()
        self.current_idx = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        task.vnf_placement = []
        task.cpu_alloc = []

        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:]

        selected_nodes = []

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            chosen_idx = int(np.argmax(placement_scores[i]))
            selected_node = mec_names[chosen_idx]
            selected_nodes.append(selected_node)

            task.vnf_placement.append(selected_node)

            node_capacity = self.mec_nodes[selected_node].cpu_capacity
            cpu_alloc = max(8.0, min(float(cpu_ratios[i]) * node_capacity, 30.0))
            task.cpu_alloc.append(cpu_alloc)

            service_time = vnf.cpu_cycles / cpu_alloc
            self.mec_nodes[selected_node].queue_load += service_time * 0.2

        delay = total_delay(task, self.graph, self.mec_nodes)
        slack = compute_slack(delay, task.deadline)

        # 純公式版 reward：對應計畫中的 delay + beta * slack
        reward = -(delay / 20.0 + self.beta * slack / 20.0)

        info = {
            "task_id": task.task_id,
            "delay": delay,
            "deadline": task.deadline,
            "slack": slack,
            "selected_nodes": selected_nodes,
            "cpu_alloc": [float(x) for x in task.cpu_alloc],
        }

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks
        truncated = False

        if not terminated:
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info