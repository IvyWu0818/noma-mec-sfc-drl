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


class IIoTEnvV2(gym.Env):
    """
    Per-VNF action version.

    Action (10-dim):
        [vnf1_mec0, vnf1_mec1, vnf1_mec2,
         vnf2_mec0, vnf2_mec1, vnf2_mec2,
         vnf3_mec0, vnf3_mec1, vnf3_mec2,
         cpu_ratio]

    Observation (10-dim):
        [data_size, deadline, sfc_length,
         cycles_v1, cycles_v2, cycles_v3,
         q0, q1, q2, total_cycles]
    """

    metadata = {"render_modes": []}

    def __init__(self, num_tasks=10, beta=10.0, seed=42):
        super().__init__()

        self.num_tasks = num_tasks
        self.beta = beta
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        # 3 VNF * 3 MEC scores + 1 cpu ratio = 10
        self.action_space = spaces.Box(
            low=np.array([0.0] * 9 + [0.1], dtype=np.float32),
            high=np.array([1.0] * 10, dtype=np.float32),
            dtype=np.float32
        )

        # [data_size, deadline, sfc_length, c1, c2, c3, q0, q1, q2, total_cycles]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(10,),
            dtype=np.float32
        )

        self.graph = None
        self.mec_nodes = None
        self.tasks = None
        self.current_idx = 0

    def _create_mec_nodes(self):
        return {
            "mec0": MECNode("mec0", 50),
            "mec1": MECNode("mec1", 50),
            "mec2": MECNode("mec2", 50)
        }

    def _create_random_task(self, task_id):
        # 中等偏難，但不要太硬
        vnfs = [VNF(i, random.randint(10, 25)) for i in range(3)]
        sfc = SFC(vnfs)

        return Task(
            task_id=task_id,
            data_size=random.randint(32, 55),
            deadline=random.randint(18, 30),
            sfc_chain=sfc
        )

    def _generate_tasks(self):
        return [self._create_random_task(i) for i in range(self.num_tasks)]

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        vnfs = task.sfc_chain.vnfs
        total_cycles = sum(v.cpu_cycles for v in vnfs)

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

        # 前 9 維：每個 VNF 各自選 MEC
        placement_scores = action[:9].reshape(3, 3)
        cpu_ratio = float(action[9])

        selected_nodes = []

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            chosen_idx = int(np.argmax(placement_scores[i]))
            selected_node = mec_names[chosen_idx]
            selected_nodes.append(selected_node)

            task.vnf_placement.append(selected_node)

            # 每段 VNF 都分配 CPU
            cpu_alloc = max(8.0, min(cpu_ratio * self.mec_nodes[selected_node].cpu_capacity, 30.0))
            task.cpu_alloc.append(cpu_alloc)

            service_time = vnf.cpu_cycles / cpu_alloc
            self.mec_nodes[selected_node].queue_load += service_time * 0.2

        delay = total_delay(task, self.graph, self.mec_nodes)
        slack = compute_slack(delay, task.deadline)

        # 負載平衡懲罰：三段 VNF 所在節點 queue 平均
        avg_selected_queue = float(np.mean([
            self.mec_nodes[node].queue_load for node in selected_nodes
        ]))
        balance_penalty = avg_selected_queue / 10.0

        # 分散部署成本：如果跨節點太多，會有額外控制成本
        unique_nodes = len(set(selected_nodes))
        split_penalty = 0.15 * (unique_nodes - 1)

        # reward 對應公式(1)，僅做 normalization
        reward = -(delay / 20.0 + self.beta * slack / 20.0 + balance_penalty + split_penalty)

        if slack == 0:
            reward += 1.0

        info = {
            "task_id": task.task_id,
            "delay": delay,
            "deadline": task.deadline,
            "slack": slack,
            "selected_nodes": selected_nodes,
            "avg_selected_queue": avg_selected_queue,
            "balance_penalty": balance_penalty,
            "split_penalty": split_penalty,
        }

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks
        truncated = False

        if not terminated:
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info