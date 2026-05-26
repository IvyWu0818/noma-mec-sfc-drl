import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode


# ═══════════════════════════════════════════════════════════════════════════
# 5G NR NOMA 系統參數（符合 3GPP TS 38.101 / TR 38.812 / TR 22.804）
# ═══════════════════════════════════════════════════════════════════════════

N_CHANNELS      = 3
MAX_NOMA_PER_CH = 2
BANDWIDTH_MHZ   = 9.0
TX_POWER_W      = 0.2
NOISE_POWER_W   = 4e-3
TIME_SLOT_MS    = 0.5

BACKHAUL_DELAY_MS = {
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

# V17: 總 MEC 算力上限（用於正規化 CPU violation → rate）
TOTAL_MEC_CPU_CAP = 35.0 + 45.0 + 55.0   # = 135.0 MCycles/ms


class IIoTEnvV17(gym.Env):
    """
    V17: 從 V14 為基礎重新設計，修正 V15/V16 的 CPU violation 惡化問題

    ═══════════════════════════════════════════════════════════════════
    V14 → V17 的改變（跳過 V15/V16，因兩版皆比 V14 差）
    ═══════════════════════════════════════════════════════════════════

    [問題診斷]
      V15: 加入 +1.5×t_link → t_link 改善但 VNF 過度集中同節點
           → CPU violation 0.21→0.48，overflow 4.65%→11.9%
      V16: 降低 t_link 係數 → CPU violation 再惡化至 1.19 MCycles/ms
           → reward 從 -21.54 衰退至 -38.22

    [改動 1] CPU Violation 改為 Rate 形式
      原: cpu_viol = raw MCycles/ms 超載量（無界，難以比較）
      新: cpu_viol_rate = cpu_viol / TOTAL_MEC_CPU_CAP ∈ [0,1]
      Reward: 900.0 × cpu_viol_rate（等效強度 ≈ V14 的 6.7 倍）

    [改動 2] Channel Overflow Penalty 加強：15.0 → 20.0
      V15/V16 overflow 從 4.65% 飆升至 11.3%，需要更強懲罰抑制

    [改動 3] Reward Scale：50 → 75
      V16 reward std = 7.26（V14 = 1.36），放大分母讓訓練更穩定

    ═══════════════════════════════════════════════════════════════════
    完整 Cost 函數（V17）:
      cost = 1.0×delay + 12.0×slack + 900.0×cpu_viol_rate
           + 0.5×t_comp + 1.5×deadline_pressure + 20.0×channel_overflow
      reward = -cost / 75.0
    ═══════════════════════════════════════════════════════════════════
    Obs 維度 (21)：同 V14
    Action 維度 (16)：同 V14
    ═══════════════════════════════════════════════════════════════════
    """

    SLOT_TASK_SIZE = N_CHANNELS * MAX_NOMA_PER_CH  # = 6 tasks per time-slot

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=75.0):
        super().__init__()
        self.num_tasks    = num_tasks
        self.beta         = beta
        self.reward_scale = reward_scale
        self.np_random    = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.zeros(16, dtype=np.float32),
            high=np.ones(16, dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-100.0, high=1000.0, shape=(21,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    def _sample_channel_gains(self) -> np.ndarray:
        h_sq = self.np_random.exponential(scale=1.0, size=N_CHANNELS).astype(np.float32)
        return np.clip(h_sq, 0.1, 2.0)

    def _compute_sinr(self, task_idx: int, ch: int) -> float:
        h_sq_u     = float(self._task_channel_gains[task_idx][ch])
        slot_start = (self.current_idx // self.SLOT_TASK_SIZE) * self.SLOT_TASK_SIZE
        interference = 0.0
        for other_idx in range(slot_start, self.current_idx):
            if self._channel_assignment[other_idx] == ch:
                h_sq_other = float(self._task_channel_gains[other_idx][ch])
                if h_sq_other < h_sq_u:
                    interference += TX_POWER_W * h_sq_other
        return float((TX_POWER_W * h_sq_u) / (interference + NOISE_POWER_W))

    def _sinr_to_rate(self, sinr: float) -> float:
        return BANDWIDTH_MHZ * np.log2(1.0 + sinr)  # Mbps

    def _slot_ch_remaining(self) -> list:
        return [
            max(0.0, (MAX_NOMA_PER_CH - self._slot_ch_count[k]) / MAX_NOMA_PER_CH)
            for k in range(N_CHANNELS)
        ]

    def _feasibility_projection(self, node_cpu_used: dict):
        """算力可行化投影（公式 5、6）。回傳 cpu_viol_rate ∈ [0,1]。"""
        mec_names = ["mec0", "mec1", "mec2"]
        cpu_viol_raw = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )
        # V17: 正規化為 rate，與 channel_overflow_ratio 量綱一致
        cpu_viol_rate = cpu_viol_raw / TOTAL_MEC_CPU_CAP

        projected = dict(node_cpu_used)
        for n in mec_names:
            cap = self.mec_nodes[n].cpu_capacity
            if projected[n] > cap:
                self._last_cpu_scale[n] = cap / projected[n]
                projected[n] = cap
            else:
                self._last_cpu_scale[n] = 1.0
        return projected, cpu_viol_rate

    def _sample_task_regime(self) -> int:
        p = float(self.np_random.random())
        if p < 0.35:   return 0
        elif p < 0.70: return 1
        else:          return 2

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
        task      = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]
        total_c   = float(sum(v.cpu_cycles for v in task.sfc_chain.vnfs))

        mec_rem = [
            max(0.0,
                (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]
        queue_load_abs  = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        queue_load_norm = [
            float(self.mec_nodes[n].queue_load / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]
        pressure  = (task.data_size + total_c) / max(task.deadline, 1)
        ch_rem    = self._slot_ch_remaining()
        queue_delta = [float(queue_load_abs[i] - self._prev_queue[i]) for i in range(3)]
        best_sinr   = float(np.max(self._task_channel_gains[self.current_idx]))

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],
            *mec_rem,
            *queue_load_norm,
            total_c,
            float(pressure),
            best_sinr,
            float(getattr(task, "task_type_id", 0)),
            *ch_rem,
            *queue_delta,
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────
    # Gym API
    # ────────────────────────────────────────────────────────────────

    def step(self, action):
        task      = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # ── 1. 部分卸載比例 ρ_u ────────────────────────────────────
        rho = float(np.clip(action[15], 0.01, 1.0))

        # ── 2. NOMA 子通道指派 ──────────────────────────────────────
        channel_scores = action[12:15]
        preferred_ch   = int(np.argmax(channel_scores))

        channel_overflow = 0.0
        if self._slot_ch_count[preferred_ch] < MAX_NOMA_PER_CH:
            assigned_ch = preferred_ch
        else:
            assigned_ch      = int(np.argmin(self._slot_ch_count))
            channel_overflow = 1.0

        self._channel_assignment[self.current_idx] = assigned_ch
        self._slot_ch_count[assigned_ch] += 1

        # ── 3. NOMA 上行速率 ────────────────────────────────────────
        sinr = self._compute_sinr(self.current_idx, assigned_ch)
        ru_k = self._sinr_to_rate(sinr)
        if channel_overflow:
            ru_k *= 0.25           # SIC 失敗 → rate 降為 1/4（同 V14）
        t_ul = rho * task.data_size / max(ru_k, 1e-6)

        # ── 4. VNF 放置 + CPU 分配 ─────────────────────────────────
        placement_scores  = action[:9].reshape(3, 3)
        cpu_ratios        = action[9:12]
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

        # ── 5. 算力可行化投影 ───────────────────────────────────────
        node_cpu_used, cpu_viol_rate = self._feasibility_projection(node_cpu_used_raw)

        # ── 6. t_comp / t_link 計算 ────────────────────────────────
        t_comp, t_link, prev_node = 0.0, 0.0, None
        for sel_node, f_alloc_orig, cpu_cycles in vnf_allocs:
            f_proj  = max(f_alloc_orig * self._last_cpu_scale[sel_node], 1e-6)
            t_comp += cpu_cycles / f_proj
            if prev_node is not None and prev_node != sel_node:
                t_link += BACKHAUL_DELAY_MS.get((prev_node, sel_node), 2.0)
            prev_node = sel_node

        # ── 7. 端到端延遲 ───────────────────────────────────────────
        delay = t_ul + t_comp + t_link

        # ── 8. Slack & 違規 ────────────────────────────────────────
        slack             = max(0.0, delay - task.deadline)
        deadline_pressure = delay / max(task.deadline, 1e-6)

        # ── 9. Reward ──────────────────────────────────────────────
        cost = (
            1.0         * delay
            + self.beta  * slack
            + 900.0      * cpu_viol_rate      # V17: rate 形式，強化懲罰
            + 0.5        * t_comp
            + 1.5        * deadline_pressure
            + 20.0       * channel_overflow   # V17: 15.0 → 20.0
        )
        reward = -cost / self.reward_scale

        # ── 11. 佇列更新 ───────────────────────────────────────────
        self._prev_queue = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.65 + node_cpu_used[n] * 0.35
            )

        # ── 12. Time-slot 切換 ─────────────────────────────────────
        self.current_idx += 1
        if self.current_idx % self.SLOT_TASK_SIZE == 0:
            self._slot_ch_count = [0] * N_CHANNELS

        # ── 13. 時變通道增益更新 ───────────────────────────────────
        prev_idx = self.current_idx - 1
        self._task_channel_gains[prev_idx] = self._sample_channel_gains()

        terminated = self.current_idx >= self.num_tasks
        obs = self._get_obs() if not terminated else np.zeros(21, dtype=np.float32)

        return obs, float(reward), terminated, False, {
            "delay":             float(delay),
            "slack":             float(slack),
            "cpu_viol_rate":     float(cpu_viol_rate),   # V17: rate [0,1]
            "t_ul":              float(t_ul),
            "t_comp":            float(t_comp),
            "t_link":            float(t_link),
            "deadline_pressure": float(deadline_pressure),
            "task_type_id":      int(getattr(task, "task_type_id", 0)),
            "assigned_ch":       int(assigned_ch),
            "channel_overflow":  float(channel_overflow),
            "sinr":              float(sinr),
            "sinr_db":           float(10 * np.log10(max(sinr, 1e-10))),
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
        self._slot_ch_count      = [0] * N_CHANNELS
        self._last_cpu_scale     = {"mec0": 1.0, "mec1": 1.0, "mec2": 1.0}
        self._prev_queue         = [0.0, 0.0, 0.0]
        self.current_idx         = 0
        return self._get_obs(), {}
