import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode


# -----------------------------------------------------------------------
# NOMA 系統參數
# -----------------------------------------------------------------------
N_CHANNELS = 3          # 子通道數量 K (公式 2-3)
MAX_NOMA_PER_CH = 2     # 每通道最多同時傳輸任務數 M (公式 3)
BANDWIDTH_MHZ = 9.0     # 每通道頻寬 B_k (公式 10)
NOISE_POWER = 0.01      # 雜訊功率 N_0 (公式 10)
TX_POWER = 0.5          # 固定發射功率 P_u (NOMA 規則固定，不作為決策變數)


class IIoTEnvV9(gym.Env):
    """
    V9: 正式加入 NOMA 子通道指派決策

    V8 → V9 的核心改變：
    ─────────────────────────────────────────────────────────────────
    [Action] 從 12 維擴充至 15 維：
      [0:9]   placement scores (3 VNFs × 3 MECs)   ← 同 V7/V8
      [9:12]  cpu_ratios (per-VNF)                  ← 同 V7/V8
      [12:15] channel_scores (N_CHANNELS=3)          ← NEW：子通道指派
              argmax → 選定子通道 k，對齊論文公式 (2)

    [Obs] 從 15 維擴充至 18 維 (+3)：
      原 V8 的 15 維 + 每個子通道的剩餘容量 ch_rem[0..2]
      ch_rem[k] = max(0, M - 目前該通道已分配任務數) / M
      讓 agent 可見通道壅塞狀態 → 對齊論文 H_t

    [NOMA 速率] 公式 (10) 完整實作：
      R_{u,k} = B_k * log2(1 + P_u * h_{u,k} / (Σ_{v∈I_k(u)} P_v*h_{v,k} + N_0))
      h_{u,k} 為 per-task per-channel 通道增益（隨機取樣模擬時變通道）
      I_k(u)  為同通道同時傳輸的其他任務集合（按 SIC 解碼順序）

    [限制式] 每通道最多 M=2 個任務 (公式 3)：
      若 agent 選擇的通道已滿，自動 fallback 到剩餘容量最大的通道
      並在 reward 中加入輕微 channel_overflow 懲罰

    [Reward] 同 V8 結構 + channel_overflow 軟懲罰項：
      cost = 1.0*delay + beta*slack + 5.0*cpu_viol
             + 0.5*t_comp + 1.5*deadline_pressure
             + 3.0*channel_overflow

    Obs 維度 (18)：
      data_size, deadline,
      cycles_v1, cycles_v2, cycles_v3,
      mec_rem0, mec_rem1, mec_rem2,
      q0, q1, q2,
      total_c, pressure, sinr,
      task_type_id,
      ch_rem0, ch_rem1, ch_rem2      ← NEW (3 dims)
    """

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=50.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta
        self.reward_scale = reward_scale
        self.np_random = np.random.default_rng(seed)

        # ── Action Space ──────────────────────────────────────────
        # [0:9]   placement scores
        # [9:12]  cpu_ratios
        # [12:15] channel_scores  ← NEW
        self.action_space = spaces.Box(
            low=np.array([0.0] * 12 + [0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0] * 15, dtype=np.float32),
            dtype=np.float32
        )

        # ── Observation Space (18 dims) ───────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1000.0, shape=(18,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    def _sample_channel_gains(self) -> np.ndarray:
        """
        取樣 per-task per-channel 通道增益 h_{u,k}。
        模擬瑞利衰落：h ~ Exp(1)，範圍 [0.1, 2.0] clip 保穩定。
        """
        gains = self.np_random.exponential(scale=1.0, size=N_CHANNELS).astype(np.float32)
        return np.clip(gains, 0.1, 2.0)

    def _compute_sinr(self, task_idx: int, ch: int) -> float:
        """
        公式 (10) 完整 SINR 計算。
        I_k(u) = 同通道中已排在 u 之前（增益更高）的任務集合（SIC 順序）。
        V9 簡化：同通道中 h 較大者先解碼，u 面對 h 較小者的干擾。
        """
        h_u = float(self._task_channel_gains[task_idx][ch])
        # 找同通道其他任務
        interferers_h = []
        for other_idx, other_ch in enumerate(self._channel_assignment):
            if other_ch == ch and other_idx != task_idx:
                h_other = float(self._task_channel_gains[other_idx][ch])
                # SIC：h_other < h_u 的任務是干擾源（尚未被消去）
                if h_other < h_u:
                    interferers_h.append(h_other)

        interference = sum(TX_POWER * h for h in interferers_h)
        sinr = (TX_POWER * h_u) / (interference + NOISE_POWER)
        return float(sinr)

    def _sinr_to_rate(self, sinr: float) -> float:
        """公式 (10): R_{u,k} = B_k * log2(1 + SINR)"""
        return BANDWIDTH_MHZ * np.log2(1.0 + sinr)

    def _sample_task_regime(self) -> int:
        p = float(self.np_random.random())
        if p < 0.35:
            return 0   # urgent
        elif p < 0.70:
            return 1   # compute-heavy
        else:
            return 2   # bandwidth-heavy

    def _build_task_by_type(self, task_id: int, task_type: int) -> Task:
        rng = random.Random(int(self.np_random.integers(0, 2**31)))
        if task_type == 0:       # urgent
            data_size = rng.randint(18, 32)
            deadline  = rng.randint(10, 16)
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]
        elif task_type == 1:     # compute-heavy
            data_size = rng.randint(25, 40)
            deadline  = rng.randint(14, 22)
            vnfs = [VNF(j, rng.randint(18, 30)) for j in range(3)]
        else:                    # bandwidth-heavy
            data_size = rng.randint(40, 65)
            deadline  = rng.randint(14, 22)
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]
        task = Task(task_id, data_size, deadline, SFC(vnfs))
        task.task_type_id = task_type
        return task

    def _channel_remaining(self) -> list:
        """每個子通道剩餘可接受任務數比例（0~1）"""
        counts = [0] * N_CHANNELS
        for ch in self._channel_assignment:
            if ch >= 0:
                counts[ch] += 1
        return [max(0.0, (MAX_NOMA_PER_CH - c) / MAX_NOMA_PER_CH) for c in counts]

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        total_c = float(sum(v.cpu_cycles for v in task.sfc_chain.vnfs))
        mec_names = ["mec0", "mec1", "mec2"]

        mec_rem = [
            max(0.0,
                (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]
        queue_load_abs = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        pressure = (task.data_size + total_c) / max(task.deadline, 1)
        ch_rem = self._channel_remaining()

        # 給 agent 看當前任務的最佳通道估計（以 h 最大通道為代理）
        current_gains = self._task_channel_gains[self.current_idx]
        best_sinr = float(np.max(current_gains))  # 簡化為最大增益通道的 SINR 代理

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],  # 3
            *mec_rem,                                              # 3
            *queue_load_abs,                                       # 3
            total_c,
            float(pressure),
            best_sinr,                                             # H_t (代理)
            float(getattr(task, "task_type_id", 0)),
            *ch_rem,                                               # 3 ← NEW
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────
    # Gym API
    # ────────────────────────────────────────────────────────────────

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # ── 1. NOMA 子通道指派 (公式 2, 3) ─────────────────────────
        channel_scores = action[12:15]                     # shape (3,)
        preferred_ch   = int(np.argmax(channel_scores))   # 公式 (2)

        ch_counts = [0] * N_CHANNELS
        for ch in self._channel_assignment:
            if ch >= 0:
                ch_counts[ch] += 1

        channel_overflow = 0.0
        if ch_counts[preferred_ch] < MAX_NOMA_PER_CH:
            assigned_ch = preferred_ch
        else:
            # Fallback：找剩餘容量最大的通道（公式 3 的軟處理）
            assigned_ch = int(np.argmin(ch_counts))
            channel_overflow = 1.0  # 懲罰 agent 沒做好通道規劃

        self._channel_assignment[self.current_idx] = assigned_ch

        # ── 2. 計算 NOMA 速率 R_{u,k} (公式 10) ────────────────────
        sinr = self._compute_sinr(self.current_idx, assigned_ch)
        ru_k = self._sinr_to_rate(sinr)
        t_ul = task.data_size / max(ru_k, 1e-6)

        # ── 3. VNF 放置 + CPU 分配 + 串接延遲 (公式 11, 12) ─────────
        t_comp = 0.0
        t_link = 0.0
        prev_node = None
        node_cpu_used = {n: 0.0 for n in mec_names}

        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:12]
        selected_nodes = []

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            c_idx    = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]
            selected_nodes.append(sel_node)

            node_cap = self.mec_nodes[sel_node].cpu_capacity
            f_min = max(
                vnf.cpu_cycles / max(task.deadline * 0.6, 1.0),
                node_cap * 0.12
            )
            f_alloc = float(np.clip(cpu_ratios[i] * node_cap, f_min, node_cap))

            t_comp += vnf.cpu_cycles / f_alloc           # 公式 (11)

            if prev_node is not None and prev_node != sel_node:
                t_link += 2.0                             # 公式 (12)

            node_cpu_used[sel_node] += f_alloc
            prev_node = sel_node

        # ── 4. 端到端延遲 (公式 13) ──────────────────────────────────
        delay = t_ul + t_comp + t_link

        # ── 5. Slack & 違規量 (公式 7-8) ────────────────────────────
        slack = max(0.0, delay - task.deadline)

        cpu_viol = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )

        deadline_pressure = delay / max(task.deadline, 1e-6)

        # ── 6. Reward (對齊論文目標式 + 輔助項) ─────────────────────
        # cost = ω1*delay + ω2*slack + ω3*cpu_viol
        #        + ω4*t_comp + ω5*deadline_pressure + ω6*channel_overflow
        cost = (
            1.0  * delay
            + self.beta * slack
            + 5.0  * cpu_viol
            + 0.5  * t_comp
            + 1.5  * deadline_pressure
            + 3.0  * channel_overflow    # 公式 (3) 違規的軟懲罰
        )
        reward = -cost / self.reward_scale

        # ── 7. 佇列更新 ──────────────────────────────────────────────
        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.65 + node_cpu_used[n] * 0.35
            )

        # ── 8. 下一步通道增益取樣（時變通道）────────────────────────
        self._task_channel_gains[self.current_idx] = self._sample_channel_gains()

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks

        obs = self._get_obs() if not terminated else np.zeros(18, dtype=np.float32)

        return obs, reward, terminated, False, {
            "delay":              delay,
            "slack":              slack,
            "cpu_viol":           cpu_viol,
            "t_ul":               t_ul,
            "t_comp":             t_comp,
            "t_link":             t_link,
            "deadline_pressure":  deadline_pressure,
            "task_type_id":       getattr(task, "task_type_id", 0),
            "assigned_ch":        assigned_ch,
            "channel_overflow":   channel_overflow,
            "sinr":               sinr,
            "ru_k":               ru_k,
            "selected_nodes":     selected_nodes,
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

        # 每個任務 × 每個通道的通道增益矩陣 h_{u,k}
        self._task_channel_gains = np.array([
            self._sample_channel_gains() for _ in range(self.num_tasks)
        ])  # shape: (num_tasks, N_CHANNELS)

        # 子通道指派記錄，-1 表示尚未指派
        self._channel_assignment = [-1] * self.num_tasks

        self.current_idx = 0
        return self._get_obs(), {}
