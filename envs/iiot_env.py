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


class IIoTEnv(gym.Env):
    """
    Minimal RL environment for IIoT + MEC + SFC scheduling.

    Action (continuous Box):
        [mec_score_0, mec_score_1, mec_score_2, cpu_ratio]

    Interpretation:
        - choose the MEC node with the largest score
        - allocate CPU based on cpu_ratio

    Observation:
        [data_size, deadline, sfc_length, total_cycles, queue_mec0, queue_mec1, queue_mec2]
    """

    metadata = {"render_modes": []}

    def __init__(self, num_tasks=10, beta=10.0, seed=42):
        super().__init__()

        self.num_tasks = num_tasks
        self.beta = beta
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # 3 MEC preference scores + 1 cpu ratio
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # [data_size, deadline, sfc_length, total_cycles, q0, q1, q2]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(7,),
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
        vnfs = [VNF(i, random.randint(10, 25)) for i in range(3)]
        sfc = SFC(vnfs)

        return Task(
            task_id=task_id,
            data_size=random.randint(30, 50),
            deadline=random.randint(20, 35),
            sfc_chain=sfc
        )

    def _generate_tasks(self):
        return [self._create_random_task(i) for i in range(self.num_tasks)]

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        total_cycles = sum(v.cpu_cycles for v in task.sfc_chain.vnfs)

        obs = np.array([
            float(task.data_size),
            float(task.deadline),
            float(len(task.sfc_chain.vnfs)),
            float(total_cycles),
            float(self.mec_nodes["mec0"].queue_load),
            float(self.mec_nodes["mec1"].queue_load),
            float(self.mec_nodes["mec2"].queue_load),
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

        mec_scores = action[:3]
        cpu_ratio = float(action[3])

        mec_names = ["mec0", "mec1", "mec2"]
        selected_node = mec_names[int(np.argmax(mec_scores))]

        task.vnf_placement = []
        task.cpu_alloc = []

        for vnf in task.sfc_chain.vnfs:
            task.vnf_placement.append(selected_node)

            cpu_alloc = max(8.0, min(cpu_ratio * self.mec_nodes[selected_node].cpu_capacity, 30.0))
            task.cpu_alloc.append(cpu_alloc)

            service_time = vnf.cpu_cycles / cpu_alloc
            self.mec_nodes[selected_node].queue_load += service_time * 0.2

        delay = total_delay(task, self.graph, self.mec_nodes)
        slack = compute_slack(delay, task.deadline)

        reward = -(delay + self.beta * slack)

        info = {
            "task_id": task.task_id,
            "delay": delay,
            "deadline": task.deadline,
            "slack": slack,
            "selected_node": selected_node,
        }

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks
        truncated = False

        if not terminated:
            obs = self._get_obs()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info