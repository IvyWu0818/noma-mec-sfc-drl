import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 假設你原本的 core 模組結構不變
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology

class IIoTEnvV4(gym.Env):
    """
    V4: 完全對齊論文 MINLP 建模
    1. 加入 NOMA 傳輸延遲模擬 (公式 9, 10)
    2. 加入 VNF 跨節點串接延遲 (公式 12)
    3. 強化 Observation: 包含節點剩餘算力與時限壓力 (公式 84)
    """
    def __init__(self, num_tasks=100, beta=5.0, seed=42, timeout_penalty=15.0, cpu_violation_penalty=10.0, reward_scale=30.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta  # 論文中的 λ [cite: 48]
        self.timeout_penalty = timeout_penalty
        self.cpu_violation_penalty = cpu_violation_penalty
        self.reward_scale = reward_scale
        
        # Action: 9 placement scores + 3 cpu ratios [cite: 94]
        self.action_space = spaces.Box(
            low=np.array([0.0]*9 + [0.1, 0.1, 0.1], dtype=np.float32),
            high=np.array([1.0]*12, dtype=np.float32),
            dtype=np.float32
        )

        # Obs (11 dims): [data, deadline, v1, v2, v3, mec0_rem, mec1_rem, mec2_rem, total_c, pressure, progress] [cite: 85]
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(11,), dtype=np.float32)

        self.seed = seed
        self.reset(seed=seed)

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        vnfs = task.sfc_chain.vnfs
        total_cycles = sum(v.cpu_cycles for v in vnfs)

        # 論文 T_t: 剩餘時限壓力概念 [cite: 90]
        deadline_pressure = (task.data_size + total_cycles) / max(task.deadline, 1)

        # 節點可用資源觀測 [cite: 87]
        mec_rem = []
        for name in ["mec0", "mec1", "mec2"]:
            node = self.mec_nodes[name]
            rem = (node.cpu_capacity - node.queue_load) / node.cpu_capacity
            mec_rem.append(max(0.0, rem))

        obs = np.array([
            float(task.data_size),
            float(task.deadline),
            float(vnfs[0].cpu_cycles),
            float(vnfs[1].cpu_cycles),
            float(vnfs[2].cpu_cycles),
            *mec_rem,
            float(total_cycles),
            float(deadline_pressure),
            float(self.current_idx / self.num_tasks)
        ], dtype=np.float32)
        return obs

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]
        
        # --- 1. NOMA 上傳延遲 (公式 9, 10) [cite: 67, 70] ---
        # 模擬論文中的 Ru,k，考慮隨機干擾
        interference = np.random.uniform(0.1, 0.5)
        ru_k = 10.0 * np.log2(1 + (0.8 / (interference + 0.01))) 
        t_ul = task.data_size / ru_k

        # --- 2. 計算與串接延遲 (公式 11, 12) [cite: 72, 75] ---
        t_comp = 0
        t_link = 0
        prev_node = None
        node_cpu_used = {name: 0.0 for name in mec_names}
        
        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:]

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            chosen_idx = int(np.argmax(placement_scores[i]))
            selected_node = mec_names[chosen_idx]
            
            # 分配算力 f_u,i,e
            node_cap = self.mec_nodes[selected_node].cpu_capacity
            f_alloc = max(5.0, min(float(cpu_ratios[i]) * node_cap, 30.0))
            
            t_comp += vnf.cpu_cycles / f_alloc
            
            # 串接延遲 de,e'bh: 若跨節點則增加成本 [cite: 76]
            if prev_node is not None and prev_node != selected_node:
                t_link += 1.5 
            
            node_cpu_used[selected_node] += f_alloc
            prev_node = selected_node

        # --- 3. 總延遲與 Reward (論文目標式 1) [cite: 47, 78] ---
        delay = t_ul + t_comp + t_link
        slack = max(0.0, delay - task.deadline) # s_u [cite: 64]
        
        cpu_violation = sum(max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity) for n in mec_names)

        # Cost = Delay + beta * Slack + penalties [cite: 48, 97]
        cost = (0.6 * delay + 0.4 * self.beta * slack + 
                self.cpu_violation_penalty * cpu_violation)
        
        reward = -cost / self.reward_scale

        # 更新環境狀態 (讓下一個任務看到當前擁塞) [cite: 88]
        for n in mec_names:
            self.mec_nodes[n].queue_load = (self.mec_nodes[n].queue_load * 0.7) + (node_cpu_used[n] * 0.2)

        info = {"delay": delay, "slack": slack, "cpu_violation": cpu_violation}
        
        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape)
        
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mec_nodes = {"mec0": MECNode("mec0", 40), "mec1": MECNode("mec1", 55), "mec2": MECNode("mec2", 70)}
        self.tasks = [self._create_random_task(i) for i in range(self.num_tasks)]
        self.current_idx = 0
        return self._get_obs(), {}

    def _create_random_task(self, tid):
        vnfs = [VNF(i, random.randint(10, 20)) for i in range(3)]
        return Task(tid, random.randint(25, 45), random.randint(20, 35), SFC(vnfs))