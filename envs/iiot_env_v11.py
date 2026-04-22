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
MAX_NOMA_PER_CH = 2     # 每 time-slot 每通道最多同時傳輸任務數 M (公式 3)
BANDWIDTH_MHZ = 9.0     # 每通道頻寬 B_k (公式 10)
NOISE_POWER = 0.01      # 雜訊功率 N_0 (公式 10)
TX_POWER = 0.5          # 固定發射功率 P_u

# Time-slot 長度 (ms) — 每個 slot 內同時傳輸的任務群組上限 M
# 論文公式 (3) 約束的是「同一 time-slot 內共用子通道的裝置數」
TIME_SLOT_MS = 5.0      # 每 time-slot 5ms，約等於 t_ul 數量級

# 異質節點串接成本矩陣 d_{e,e'}^bh (ms) — 論文公式 (12)
BACKHAUL_DELAY = {
    ("mec0", "mec0"): 0.0,
    ("mec0", "mec1"): 1.5,
    ("mec0", "mec2"): 3.0,
    ("mec1", "mec0"): 1.5,
    ("mec1", "mec1"): 0.0,
    ("mec1", "mec2"): 2.0,
    ("mec2", "mec0"): 3.0,
    ("mec2", "mec1"): 2.0,
    ("mec2", "mec2"): 0.0,
}


class IIoTEnvV11(gym.Env):
    """
    V11: 修復 Channel Overflow 定義，對齊論文公式 (3) 語意

    ═══════════════════════════════════════════════════════════════════
    V10 → V11 的核心改變
    ═══════════════════════════════════════════════════════════════════

    [修復] Channel Overflow 定義（論文公式 3 的正確語意）

      V9/V10 的問題：
        channel_assignment 是 episode 層級的累積陣列，100 個任務跑完
        至少 94 個會 overflow（因為只有 K×M = 6 個 slot 不衝突），
        overflow ratio 永遠貼近理論下限 0.94，懲罰訊號失效。

      V11 的修法：
        引入 time-slot 概念。每個 slot 維護一個「當前正在傳輸的通道
        佔用計數 _slot_ch_count」，每處理 SLOT_TASK_SIZE 個任務後
        自動清空（模擬 slot 結束、新 slot 開始）。

        overflow 的判定變成：
          「在當前 time-slot 內，該通道已有 M 個任務在傳輸嗎？」

        這樣在 slot 容量足夠時，agent 完全可以避免 overflow；
        懲罰訊號才有意義，agent 才能真正學到通道規劃。

      SLOT_TASK_SIZE = MAX_NOMA_PER_CH × N_CHANNELS = 6
        每 6 個任務為一個 time-slot（每個通道剛好塞滿 M=2 個任務）
        100 個任務 → 約 17 個 slot，overflow 完全可被 agent 控制。

    [調整] ch_rem obs 語意也對應更新
        從「episode 累積剩餘」→「當前 slot 剩餘容量」
        讓 agent 觀測到真正有決策意義的通道狀態。

    [調整] channel_overflow reward 從 8.0 → 5.0
        定義修正後懲罰訊號已有意義，不需要超高懲罰來強迫行為。
        過高懲罰反而讓 agent 為了避免 overflow 而不計代價地使用其他通道。

    ═══════════════════════════════════════════════════════════════════
    Obs 維度 (21)：同 V10，但 ch_rem 語意改為 per-slot 剩餘
      data_size, deadline,
      cycles_v1, cycles_v2, cycles_v3,
      mec_rem0, mec_rem1, mec_rem2,
      q0, q1, q2,
      total_c, pressure, sinr,
      task_type_id,
      ch_rem0, ch_rem1, ch_rem2,    (per-slot 剩餘容量)
      dq0, dq1, dq2                 (佇列變化率)

    Action 維度 (16)：同 V10
      [0:9]   placement_scores
      [9:12]  cpu_ratios
      [12:15] channel_scores
      [15]    rho
    """

    # 每個 time-slot 包含的任務數 = K × M（每通道剛好填滿）
    SLOT_TASK_SIZE = N_CHANNELS * MAX_NOMA_PER_CH  # = 6

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=50.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta
        self.reward_scale = reward_scale
        self.np_random = np.random.default_rng(seed)

        # Action Space (16 維)
        self.action_space = spaces.Box(
            low=np.zeros(16, dtype=np.float32),
            high=np.ones(16, dtype=np.float32),
            dtype=np.float32
        )

        # Observation Space (21 維)
        self.observation_space = spaces.Box(
            low=-100.0, high=1000.0, shape=(21,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    def _sample_channel_gains(self) -> np.ndarray:
        """瑞利衰落通道增益 h_{u,k} ~ Exp(1)，clip [0.1, 2.0]"""
        gains = self.np_random.exponential(scale=1.0, size=N_CHANNELS).astype(np.float32)
        return np.clip(gains, 0.1, 2.0)

    def _compute_sinr(self, task_idx: int, ch: int) -> float:
        """
        公式 (10) SINR 計算。
        干擾來自同 slot 同通道、h 較小（先被消去後的殘餘）的任務。
        """
        h_u = float(self._task_channel_gains[task_idx][ch])
        # 同 slot 內同通道的其他任務
        slot_start = (self.current_idx // self.SLOT_TASK_SIZE) * self.SLOT_TASK_SIZE
        interferers_h = []
        for other_idx in range(slot_start, self.current_idx):
            if self._channel_assignment[other_idx] == ch:
                h_other = float(self._task_channel_gains[other_idx][ch])
                if h_other < h_u:
                    interferers_h.append(h_other)

        interference = sum(TX_POWER * h for h in interferers_h)
        sinr = (TX_POWER * h_u) / (interference + NOISE_POWER)
        return float(sinr)

    def _sinr_to_rate(self, sinr: float) -> float:
        """公式 (10): R_{u,k} = B_k * log2(1 + SINR)"""
        return BANDWIDTH_MHZ * np.log2(1.0 + sinr)

    def _slot_ch_remaining(self) -> list:
        """
        當前 time-slot 內各通道的剩餘容量比例。
        這才是論文公式 (3) 真正約束的狀態。
        """
        return [
            max(0.0, (MAX_NOMA_PER_CH - self._slot_ch_count[k]) / MAX_NOMA_PER_CH)
            for k in range(N_CHANNELS)
        ]

    def _feasibility_projection(self, node_cpu_used: dict):
        """算力可行化投影（公式 5、6）"""
        mec_names = ["mec0", "mec1", "mec2"]
        cpu_viol_before = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )
        projected = dict(node_cpu_used)
        for n in mec_names:
            cap = self.mec_nodes[n].cpu_capacity
            if projected[n] > cap:
                self._last_cpu_scale[n] = cap / projected[n]
                projected[n] = cap
            else:
                self._last_cpu_scale[n] = 1.0
        return projected, cpu_viol_before

    def _sample_task_regime(self) -> int:
        p = float(self.np_random.random())
        if p < 0.35:   return 0   # urgent
        elif p < 0.70: return 1   # compute-heavy
        else:          return 2   # bandwidth-heavy

    def _build_task_by_type(self, task_id: int, task_type: int) -> Task:
        rng = random.Random(int(self.np_random.integers(0, 2**31)))
        if task_type == 0:
            data_size = rng.randint(18, 32)
            deadline  = rng.randint(10, 16)
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]
        elif task_type == 1:
            data_size = rng.randint(25, 40)
            deadline  = rng.randint(14, 22)
            vnfs = [VNF(j, rng.randint(18, 30)) for j in range(3)]
        else:
            data_size = rng.randint(40, 65)
            deadline  = rng.randint(14, 22)
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]
        task = Task(task_id, data_size, deadline, SFC(vnfs))
        task.task_type_id = task_type
        return task

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]
        total_c = float(sum(v.cpu_cycles for v in task.sfc_chain.vnfs))

        mec_rem = [
            max(0.0,
                (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]
        queue_load_abs = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        pressure = (task.data_size + total_c) / max(task.deadline, 1)

        # V11: per-slot 剩餘容量（對齊公式 3 的真實語意）
        ch_rem = self._slot_ch_remaining()

        queue_delta = [
            float(queue_load_abs[i] - self._prev_queue[i])
            for i in range(3)
        ]

        current_gains = self._task_channel_gains[self.current_idx]
        best_sinr = float(np.max(current_gains))

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],
            *mec_rem,
            *queue_load_abs,
            total_c,
            float(pressure),
            best_sinr,
            float(getattr(task, "task_type_id", 0)),
            *ch_rem,        # per-slot 剩餘（V11 修正語意）
            *queue_delta,
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────
    # Gym API
    # ────────────────────────────────────────────────────────────────

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # ── 1. 部分卸載比例 ρ_u ────────────────────────────────────
        rho = float(np.clip(action[15], 0.01, 1.0))

        # ── 2. NOMA 子通道指派（公式 2、3）────────────────────────
        # V11: 用 per-slot 計數判斷 overflow（真正對齊公式 3）
        channel_scores = action[12:15]
        preferred_ch   = int(np.argmax(channel_scores))

        channel_overflow = 0.0
        if self._slot_ch_count[preferred_ch] < MAX_NOMA_PER_CH:
            assigned_ch = preferred_ch
        else:
            # Fallback：找 slot 內剩餘最多的通道
            assigned_ch = int(np.argmin(self._slot_ch_count))
            channel_overflow = 1.0

        self._channel_assignment[self.current_idx] = assigned_ch
        self._slot_ch_count[assigned_ch] += 1

        # ── 3. NOMA 速率 R_{u,k} (公式 10) ────────────────────────
        sinr = self._compute_sinr(self.current_idx, assigned_ch)
        ru_k = self._sinr_to_rate(sinr)
        t_ul = rho * task.data_size / max(ru_k, 1e-6)

        # ── 4. VNF 放置 + CPU 分配 ─────────────────────────────────
        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios       = action[9:12]

        node_cpu_used_raw = {n: 0.0 for n in mec_names}
        vnf_allocs, selected_nodes = [], []

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            c_idx    = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]
            selected_nodes.append(sel_node)

            node_cap = self.mec_nodes[sel_node].cpu_capacity
            f_min    = max(
                vnf.cpu_cycles / max(task.deadline * 0.6, 1.0),
                node_cap * 0.12
            )
            f_alloc  = float(np.clip(cpu_ratios[i] * node_cap, f_min, node_cap))
            node_cpu_used_raw[sel_node] += f_alloc
            vnf_allocs.append((sel_node, f_alloc, vnf.cpu_cycles))

        # ── 5. 算力可行化投影（公式 5、6）──────────────────────────
        node_cpu_used, cpu_viol = self._feasibility_projection(node_cpu_used_raw)

        t_comp, t_link, prev_node = 0.0, 0.0, None
        for sel_node, f_alloc_orig, cpu_cycles in vnf_allocs:
            f_proj  = max(f_alloc_orig * self._last_cpu_scale[sel_node], 1e-6)
            t_comp += cpu_cycles / f_proj
            if prev_node is not None and prev_node != sel_node:
                t_link += BACKHAUL_DELAY.get((prev_node, sel_node), 2.0)
            prev_node = sel_node

        # ── 6. 端到端延遲（公式 13）───────────────────────────────
        delay = t_ul + t_comp + t_link

        # ── 7. Slack & 違規 ────────────────────────────────────────
        slack             = max(0.0, delay - task.deadline)
        deadline_pressure = delay / max(task.deadline, 1e-6)

        # ── 8. Reward ──────────────────────────────────────────────
        # channel_overflow 懲罰從 8.0 → 5.0：
        # 定義修正後懲罰訊號已有意義，不再需要超高懲罰
        cost = (
            1.0        * delay
            + self.beta * slack
            + 5.0       * cpu_viol
            + 0.5       * t_comp
            + 1.5       * deadline_pressure
            + 5.0       * channel_overflow   # 從 8.0 降回合理值
        )
        reward = -cost / self.reward_scale

        # ── 9. 佇列更新 ────────────────────────────────────────────
        self._prev_queue = [
            float(self.mec_nodes[n].queue_load) for n in mec_names
        ]
        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.65 + node_cpu_used[n] * 0.35
            )

        # ── 10. Time-slot 切換：每 SLOT_TASK_SIZE 個任務重置 slot ──
        self.current_idx += 1
        if self.current_idx % self.SLOT_TASK_SIZE == 0:
            self._slot_ch_count = [0] * N_CHANNELS  # 新 slot 開始

        # ── 11. 時變通道增益更新 ───────────────────────────────────
        prev_idx = self.current_idx - 1
        self._task_channel_gains[prev_idx] = self._sample_channel_gains()

        terminated = self.current_idx >= self.num_tasks
        obs = self._get_obs() if not terminated else np.zeros(21, dtype=np.float32)

        return obs, float(reward), terminated, False, {
            "delay":             float(delay),
            "slack":             float(slack),
            "cpu_viol":          float(cpu_viol),
            "t_ul":              float(t_ul),
            "t_comp":            float(t_comp),
            "t_link":            float(t_link),
            "deadline_pressure": float(deadline_pressure),
            "task_type_id":      int(getattr(task, "task_type_id", 0)),
            "assigned_ch":       int(assigned_ch),
            "channel_overflow":  float(channel_overflow),
            "sinr":              float(sinr),
            "ru_k":              float(ru_k),
            "rho":               float(rho),
            "selected_nodes":    selected_nodes,
            "slot_id":           int((self.current_idx - 1) // self.SLOT_TASK_SIZE),
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

        self._task_channel_gains = np.array([
            self._sample_channel_gains() for _ in range(self.num_tasks)
        ])

        self._channel_assignment = [-1] * self.num_tasks

        # V11 核心：per-slot 通道計數（每 SLOT_TASK_SIZE 個任務重置）
        self._slot_ch_count = [0] * N_CHANNELS

        self._last_cpu_scale = {"mec0": 1.0, "mec1": 1.0, "mec2": 1.0}
        self._prev_queue     = [0.0, 0.0, 0.0]

        self.current_idx = 0
        return self._get_obs(), {}
