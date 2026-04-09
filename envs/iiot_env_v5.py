import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode

class IIoTEnvV5(gym.Env):
    """
    V5: 動態高壓環境測試
    - 任務時限 (Deadline) 縮短 20%，模擬緊急工業控制需求 [cite: 40]
    - NOMA 干擾範圍擴大，模擬高連線密度 [cite: 25]
    - 加重超時懲罰 Lambda，對齊論文目標式 (1) [cite: 47]
    """
    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=30.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta  # 論文中的 λ [cite: 48]
        self.reward_scale = reward_scale
        
        # Action: 9 placement + 3 cpu ratios [cite: 94]
        self.action_space = spaces.Box(
            low=np.array([0.0]*9 + [0.1, 0.1, 0.1], dtype=np.float32),
            high=np.array([1.0]*12, dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(11,), dtype=np.float32)
        self.reset(seed=seed)

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        total_c = sum(v.cpu_cycles for v in task.sfc_chain.vnfs)
        # 觀測各 MEC 剩餘算力百分比 [cite: 87]
        mec_rem = [(self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load) / self.mec_nodes[n].cpu_capacity for n in ["mec0", "mec1", "mec2"]]
        # 論文 T_t: 剩餘時限壓力 [cite: 90]
        pressure = (task.data_size + total_c) / max(task.deadline, 1)

        return np.array([
            float(task.data_size), float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],
            *[max(0.0, float(r)) for r in mec_rem],
            float(total_c), float(pressure), float(self.current_idx / self.num_tasks)
        ], dtype=np.float32)

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]
        
        # 1. 時變 NOMA 速率 Ru,k (公式 10) [cite: 70, 85]
        interference = np.random.uniform(0.15, 0.6) # 提高干擾下限
        ru_k = 9.0 * np.log2(1 + (0.5 / (interference + 0.01))) 
        t_ul = task.data_size / ru_k

        # 2. 聯合編排 (公式 11, 12) [cite: 73, 75]
        t_comp, t_link = 0, 0
        prev_node = None
        node_cpu_used = {name: 0.0 for name in mec_names}
        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:]

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            c_idx = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]
            # 分配算力，模擬更嚴苛的資源上限 [cite: 60]
            f_alloc = max(5.0, min(float(cpu_ratios[i]) * self.mec_nodes[sel_node].cpu_capacity, 25.0))
            t_comp += vnf.cpu_cycles / f_alloc
            # 串接成本 de,e'bh [cite: 76]
            if prev_node and prev_node != sel_node: t_link += 2.0 
            node_cpu_used[sel_node] += f_alloc
            prev_node = sel_node

        # 3. 延遲與 Reward (公式 13, 97) [cite: 78, 97]
        delay = t_ul + t_comp + t_link
        slack = max(0.0, delay - task.deadline) # s_u [cite: 64]
        cpu_viol = sum(max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity) for n in mec_names)

        # 加重 Slack 權重 beta，對齊論文 [cite: 48, 97]
        cost = 0.5 * delay + self.beta * slack + 15.0 * cpu_viol
        reward = -cost / self.reward_scale

        # 更新佇列負載，增加積壓感 [cite: 86]
        for n in mec_names:
            self.mec_nodes[n].queue_load = (self.mec_nodes[n].queue_load * 0.6) + (node_cpu_used[n] * 0.3)

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks
        return self._get_obs() if not terminated else np.zeros(11), reward, terminated, False, {"delay": delay, "slack": slack}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mec_nodes = {"mec0": MECNode("mec0", 35), "mec1": MECNode("mec1", 45), "mec2": MECNode("mec2", 55)}
        # 更加緊迫的任務生成：資料量更大、時限更短 [cite: 40]
        self.tasks = [Task(i, random.randint(30, 55), random.randint(18, 28), SFC([VNF(j, random.randint(12, 22)) for j in range(3)])) for i in range(self.num_tasks)]
        self.current_idx = 0
        return self._get_obs(), {}