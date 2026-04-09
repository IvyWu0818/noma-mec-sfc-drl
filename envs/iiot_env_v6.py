import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode

class IIoTEnvV6(gym.Env):
    """
    V6: 修正版 — 對齊論文公式與限制式

    相較 V5 的修正：
    1. [State] 加入通道狀態 H_t (SINR 估計) — 對齊論文 s_t 定義
    2. [State] 加入佇列絕對負載 Q_t — 補充 V5 只有百分比的不足
    3. [Action/公式6] 修正 CPU 分配耦合：f_alloc 只由放置節點的 ratio 決定，
       未放置節點的 VNF 不佔用其他節點算力，對齊 z_{u,i,e} * F_e 約束
    4. [NOMA/公式10] 通道增益 h 改為 per-task 參數化，使速率與任務相關，
       並將當前 SINR 加入 observation
    5. [Reward] 維持 V5 的 cost 結構 (對齊論文目標式)，但使用正確的 per-node
       算力統計（只累計真正放置在該節點的算力）
    6. [Reset] 使用 np RNG 確保可重現性
    7. [Obs 維度] 從 11 擴充至 14：
         data_size, deadline,
         cpu_cycles[0..2],        (SFC 計算需求)
         mec_rem[0..2],           (剩餘容量 %)
         queue_load[0..2],        (佇列絕對負載) ← NEW
         total_c, pressure,       (SFC 總計算量, 時限壓力)
         sinr_est                 (當前 NOMA 通道品質估計) ← NEW
    """

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=30.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta          # 論文中的 λ，超時懲罰權重
        self.reward_scale = reward_scale
        self.np_random = np.random.default_rng(seed)

        # --- Action Space ---
        # 9 placement scores (3 VNFs × 3 MECs) + 3 cpu ratios
        # placement: argmax over dim=1 → 對齊公式 (4) z_{u,i,e} ∈ {0,1}
        # cpu_ratios: 對齊公式 (6) 0 ≤ f_{u,i,e} ≤ z_{u,i,e}·F_e
        self.action_space = spaces.Box(
            low=np.array([0.0] * 9 + [0.1, 0.1, 0.1], dtype=np.float32),
            high=np.array([1.0] * 12, dtype=np.float32),
            dtype=np.float32
        )

        # --- Observation Space ---
        # 14 維：data_size, deadline, cpu×3, mec_rem×3, queue_load×3, total_c, pressure, sinr
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(14,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_sinr(self) -> float:
        """
        模擬任務的 NOMA 上傳 SINR (公式 10 中的 P_u·h_u,k / (干擾+N0))。
        V6 改成 per-task 取樣，並回傳值以加入 observation (H_t)。
        干擾範圍對齊 V5 高壓設定 [0.15, 0.6]。
        """
        interference = self.np_random.uniform(0.15, 0.6)
        sinr = 0.5 / (interference + 0.01)   # P·h / (I + N0)，歸一化
        return float(sinr)

    def _sinr_to_rate(self, sinr: float) -> float:
        """公式 (10): R_{u,k} = B · log2(1 + SINR)，B≈9 MHz 參數化"""
        return 9.0 * np.log2(1.0 + sinr)

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        total_c = sum(v.cpu_cycles for v in task.sfc_chain.vnfs)
        mec_names = ["mec0", "mec1", "mec2"]

        # 剩餘算力百分比 — C_t (公式 5 資源約束的可觀測代理)
        mec_rem = [
            max(0.0, (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]

        # 佇列絕對負載 — Q_t (補論文 Q_t 定義)
        queue_load_abs = [
            float(self.mec_nodes[n].queue_load) for n in mec_names
        ]

        # 時限壓力 — T_t (論文 T_u^ratio 的近似，值越大超時風險越高)
        pressure = (task.data_size + total_c) / max(task.deadline, 1)

        # 通道 SINR 估計 — H_t (論文 state 成員)
        sinr = self._current_sinr  # 由上一次 step 或 reset 取樣

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],  # 3 dims
            *mec_rem,                                              # 3 dims
            *queue_load_abs,                                       # 3 dims ← NEW
            float(total_c),
            float(pressure),
            float(sinr),                                           # 1 dim ← NEW
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # --- 1. NOMA 上傳延遲 (公式 9, 10) ---
        # 使用本步驟的 SINR（已在 obs 中告知 agent）
        ru_k = self._sinr_to_rate(self._current_sinr)
        t_ul = task.data_size / max(ru_k, 1e-6)

        # --- 2. VNF 放置 + CPU 分配 + 串接延遲 (公式 11, 12) ---
        t_comp = 0.0
        t_link = 0.0
        prev_node = None
        node_cpu_used = {name: 0.0 for name in mec_names}

        placement_scores = action[:9].reshape(3, 3)   # shape (n_vnf, n_mec)
        cpu_ratios = action[9:]                        # shape (n_vnf,)

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            # 公式 (4): z_{u,i,e} — argmax 保證唯一放置
            c_idx = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]

            # 公式 (6) 修正：f_alloc 綁定到 sel_node 的算力，
            # 未選擇節點不累計任何負載 — 對齊 0 ≤ f ≤ z·F_e
            node_capacity = self.mec_nodes[sel_node].cpu_capacity
            f_alloc = float(cpu_ratios[i]) * node_capacity
            f_alloc = float(np.clip(f_alloc, 5.0, node_capacity * 0.8))  # 最多用 80% 防霸占

            # 公式 (11): t_cmp = C_{u,i} / f_{u,i,e}
            t_comp += vnf.cpu_cycles / f_alloc

            # 公式 (12): 串接成本 d_{e,e'}^bh (跨節點固定 2.0，同節點 0)
            if prev_node is not None and prev_node != sel_node:
                t_link += 2.0

            # 只有 sel_node 累計算力消耗 — 修正 V5 的耦合問題
            node_cpu_used[sel_node] += f_alloc
            prev_node = sel_node

        # --- 3. 端到端延遲 (公式 13) ---
        delay = t_ul + t_comp + t_link

        # --- 4. Slack & CPU 違規 (公式 7-8) ---
        slack = max(0.0, delay - task.deadline)           # s_u ≥ 0
        cpu_viol = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )

        # --- 5. Reward (對齊論文目標式 + soft penalty relaxation for cpu) ---
        # cost = ω1·delay + ω2·slack + ω3·cpu_viol
        cost = 0.5 * delay + self.beta * slack + 15.0 * cpu_viol
        reward = -cost / self.reward_scale

        # --- 6. 佇列更新 ---
        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.6 + node_cpu_used[n] * 0.3
            )

        # --- 7. 取樣下一步通道狀態 (agent 在下一步 obs 中可觀測) ---
        self._current_sinr = self._sample_sinr()

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks

        obs = self._get_obs() if not terminated else np.zeros(14, dtype=np.float32)
        return obs, reward, terminated, False, {"delay": delay, "slack": slack, "cpu_viol": cpu_viol}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # MEC 節點初始化
        self.mec_nodes = {
            "mec0": MECNode("mec0", 35),
            "mec1": MECNode("mec1", 45),
            "mec2": MECNode("mec2", 55),
        }

        # 任務生成：高壓設定 (資料量大、時限緊)
        rng = random.Random(int(self.np_random.integers(0, 2**31)))
        self.tasks = [
            Task(
                i,
                rng.randint(30, 55),   # data_size
                rng.randint(18, 28),   # deadline
                SFC([VNF(j, rng.randint(12, 22)) for j in range(3)])
            )
            for i in range(self.num_tasks)
        ]

        self.current_idx = 0

        # 初始化通道狀態 (第一步 obs 可觀測)
        self._current_sinr = self._sample_sinr()

        return self._get_obs(), {}
