import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode

class IIoTEnvV7(gym.Env):
    """
    V7 (修正版): 自適應 f_alloc 下限 + t_comp 獨立梯度項

    V6 → V7 修正：
    1. [f_alloc] 自適應下限：max(cpu_cycles / deadline, node_cap × 0.15)
       確保 agent 初期有合理計算速度，避免壓低 ratio 導致 t_comp 爆炸
    2. [f_alloc] 上限 = node_capacity（軟約束由 cpu_viol penalty 控制）
    3. [Reward] 加入 -t_comp 獨立分量 (ω4=0.5)，給 t_comp 直接梯度來源，
       防止 agent 用低 cpu_ratio 規避 cpu_viol 但推高計算延遲
    4. [Reward] ω1=1.0·delay + ω2=12.0·slack + ω3=5.0·cpu_viol + ω4=0.5·t_comp
       reward_scale=50.0
    5. [Obs] 維持 V6 的 14 維不變
    """

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=50.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta
        self.reward_scale = reward_scale
        self.np_random = np.random.default_rng(seed)

        # Action: 9 placement scores + 3 cpu_ratios
        self.action_space = spaces.Box(
            low=np.array([0.0] * 9 + [0.05, 0.05, 0.05], dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32
        )

        # Obs: 14 維
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(14,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ------------------------------------------------------------------
    def _sample_sinr(self) -> float:
        interference = float(self.np_random.uniform(0.15, 0.6))
        return 0.5 / (interference + 0.01)

    def _sinr_to_rate(self, sinr: float) -> float:
        return 9.0 * np.log2(1.0 + sinr)

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        total_c = float(sum(v.cpu_cycles for v in task.sfc_chain.vnfs))
        mec_names = ["mec0", "mec1", "mec2"]

        mec_rem = [
            max(0.0, (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]
        queue_load_abs = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        pressure = (task.data_size + total_c) / max(task.deadline, 1)

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],  # 3
            *mec_rem,                                              # 3
            *queue_load_abs,                                       # 3
            total_c,
            float(pressure),
            float(self._current_sinr),                             # H_t
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # 1. 上傳延遲 (公式 9, 10)
        ru_k = self._sinr_to_rate(self._current_sinr)
        t_ul = task.data_size / max(ru_k, 1e-6)

        # 2. VNF 放置 + CPU 分配 + 串接延遲 (公式 11, 12)
        t_comp = 0.0
        t_link = 0.0
        prev_node = None
        node_cpu_used = {n: 0.0 for n in mec_names}

        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:]

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            # 公式 (4): 唯一放置
            c_idx = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]
            node_cap = self.mec_nodes[sel_node].cpu_capacity

            # V7 自適應下限：
            # 至少能在 deadline 內跑完此 VNF，且不低於節點 15% 算力
            f_min = max(
                vnf.cpu_cycles / max(task.deadline, 1),
                node_cap * 0.15
            )
            f_alloc = float(np.clip(cpu_ratios[i] * node_cap, f_min, node_cap))

            # 公式 (11)
            t_comp += vnf.cpu_cycles / f_alloc

            # 公式 (12)
            if prev_node is not None and prev_node != sel_node:
                t_link += 2.0

            node_cpu_used[sel_node] += f_alloc
            prev_node = sel_node

        # 3. 端到端延遲 (公式 13)
        delay = t_ul + t_comp + t_link

        # 4. Slack & CPU 違規
        slack = max(0.0, delay - task.deadline)
        cpu_viol = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )

        # 5. Reward — 加入 t_comp 獨立項給直接梯度
        # cost = 1.0·delay + 12.0·slack + 5.0·cpu_viol + 0.5·t_comp
        cost = (1.0 * delay
                + self.beta * slack
                + 5.0 * cpu_viol
                + 0.5 * t_comp)
        reward = -cost / self.reward_scale

        # 6. 佇列更新
        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.6 + node_cpu_used[n] * 0.3
            )

        # 7. 取樣下一步通道
        self._current_sinr = self._sample_sinr()

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks

        obs = self._get_obs() if not terminated else np.zeros(14, dtype=np.float32)
        return obs, reward, terminated, False, {
            "delay": delay,
            "slack": slack,
            "cpu_viol": cpu_viol,
            "t_ul": t_ul,
            "t_comp": t_comp,
            "t_link": t_link,
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

        rng = random.Random(int(self.np_random.integers(0, 2**31)))
        self.tasks = [
            Task(
                i,
                rng.randint(30, 55),
                rng.randint(18, 28),
                SFC([VNF(j, rng.randint(12, 22)) for j in range(3)])
            )
            for i in range(self.num_tasks)
        ]

        self.current_idx = 0
        self._current_sinr = self._sample_sinr()
        return self._get_obs(), {}
