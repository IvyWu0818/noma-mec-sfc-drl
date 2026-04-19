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
TX_POWER = 0.5          # 固定發射功率 P_u

# 異質節點串接成本矩陣 d_{e,e'}^bh (ms) — 論文公式 (12)
# 對角線為同節點（=0），非對角線為跨節點 backhaul 延遲
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


class IIoTEnvV10(gym.Env):
    """
    V10: 對齊論文架構的全面修復版本

    V9 → V10 的核心改變：
    ─────────────────────────────────────────────────────────────────
    [修復 1] Channel Overflow 問題（V9 達 96%）
      - channel_overflow reward 懲罰從 3.0 → 8.0（強化訊號）
      - channel_scores 改用帶溫度的 Gumbel softmax 採樣
        → 避免 agent 永遠 argmax 到同一通道
      - obs 加入 episode 內累積通道計數 ch_count（歸一化）
        讓 agent 明確觀測哪個通道快滿

    [修復 2] 算力可行化投影（論文公式 5、6）
      - 在 step() 輸出前加入硬約束投影：
        若 Σ f_{u,i,e} > F_e，等比例縮小分配量
      - 保留 reward 中的 cpu_viol 軟懲罰（作為輔助訊號）
      - 確保輸出決策滿足容量限制（可行解保證）

    [新增 3] 部分卸載比例 ρ_u（論文 action space）
      - action[15] = rho ∈ [0,1]，代表任務卸載比例
      - t_ul = ρ × D_u / R_{u,k}（對齊公式 9）
      - action 維度：15 → 16

    [新增 4] 佇列變化率特徵（論文 State Q_t 飽和度預警）
      - obs 加入前一步驟與當前步驟的佇列差值 Δqueue[0..2]
      - 讓 agent 預判瓶頸節點，對齊論文 State 定義
      - obs 維度：18 → 21

    [新增 5] 異質節點串接成本矩陣（論文公式 12）
      - 固定 2ms → 異質 BACKHAUL_DELAY 矩陣
      - mec0↔mec1: 1.5ms, mec0↔mec2: 3.0ms, mec1↔mec2: 2.0ms

    ─────────────────────────────────────────────────────────────────
    Obs 維度 (21)：
      data_size, deadline,
      cycles_v1, cycles_v2, cycles_v3,
      mec_rem0, mec_rem1, mec_rem2,       (算力剩餘比例)
      q0, q1, q2,                          (佇列絕對值)
      total_c, pressure, sinr,
      task_type_id,
      ch_rem0, ch_rem1, ch_rem2,           (通道剩餘容量，V9)
      dq0, dq1, dq2                        (佇列變化率，NEW)

    Action 維度 (16)：
      [0:9]   placement_scores (3 VNFs × 3 MECs)
      [9:12]  cpu_ratios (per-VNF)
      [12:15] channel_scores (N_CHANNELS=3)
      [15]    rho (部分卸載比例, NEW)
    """

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=50.0,
                 channel_temp=1.0):
        """
        Parameters
        ----------
        channel_temp : float
            Gumbel softmax 溫度。1.0 = 正常探索；較低值趨向 argmax。
            訓練初期建議 1.0，後期可 anneal 到 0.5。
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta
        self.reward_scale = reward_scale
        self.channel_temp = channel_temp
        self.np_random = np.random.default_rng(seed)

        # ── Action Space (16 維) ───────────────────────────────────
        # [0:9]  placement scores
        # [9:12] cpu_ratios
        # [12:15] channel_scores
        # [15]   rho (卸載比例)
        self.action_space = spaces.Box(
            low=np.zeros(16, dtype=np.float32),
            high=np.ones(16, dtype=np.float32),
            dtype=np.float32
        )

        # ── Observation Space (21 維) ──────────────────────────────
        self.observation_space = spaces.Box(
            low=-100.0, high=1000.0, shape=(21,), dtype=np.float32
        )

        self.reset(seed=seed)

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    def _sample_channel_gains(self) -> np.ndarray:
        """
        取樣 per-task per-channel 通道增益 h_{u,k}。
        瑞利衰落：h ~ Exp(1)，clip 至 [0.1, 2.0]。
        """
        gains = self.np_random.exponential(scale=1.0, size=N_CHANNELS).astype(np.float32)
        return np.clip(gains, 0.1, 2.0)

    def _gumbel_softmax_sample(self, logits: np.ndarray, temp: float) -> int:
        """
        Gumbel-max 採樣，鼓勵 agent 探索不同通道。
        等價於 argmax(logits + Gumbel noise / temp)。
        temp → 0 時收斂為 argmax；temp 較大時較均勻探索。
        """
        gumbel_noise = -np.log(-np.log(
            np.clip(self.np_random.random(len(logits)), 1e-10, 1 - 1e-10)
        ))
        return int(np.argmax(logits + gumbel_noise * temp))

    def _compute_sinr(self, task_idx: int, ch: int) -> float:
        """
        公式 (10) 完整 SINR 計算。
        SIC 順序：h 較大者先解碼，h 較小者為干擾源。
        """
        h_u = float(self._task_channel_gains[task_idx][ch])
        interferers_h = []
        for other_idx, other_ch in enumerate(self._channel_assignment):
            if other_ch == ch and other_idx != task_idx:
                h_other = float(self._task_channel_gains[other_idx][ch])
                if h_other < h_u:
                    interferers_h.append(h_other)

        interference = sum(TX_POWER * h for h in interferers_h)
        sinr = (TX_POWER * h_u) / (interference + NOISE_POWER)
        return float(sinr)

    def _sinr_to_rate(self, sinr: float) -> float:
        """公式 (10): R_{u,k} = B_k * log2(1 + SINR)"""
        return BANDWIDTH_MHZ * np.log2(1.0 + sinr)

    def _channel_counts(self) -> list:
        """各通道已分配任務數（episode 累積）"""
        counts = [0] * N_CHANNELS
        for ch in self._channel_assignment:
            if ch >= 0:
                counts[ch] += 1
        return counts

    def _channel_remaining(self) -> list:
        """各通道剩餘容量比例（0~1）"""
        counts = self._channel_counts()
        return [max(0.0, (MAX_NOMA_PER_CH - c) / MAX_NOMA_PER_CH) for c in counts]

    def _feasibility_projection(self, node_cpu_used: dict) -> dict:
        """
        算力可行化投影（對齊論文公式 5、6）。
        若節點超過容量，等比例縮小各 VNF 的分配量。

        Returns
        -------
        projected : dict
            滿足容量限制的算力分配（修正後）
        cpu_viol_before : float
            投影前的違規量（用於 reward 輔助懲罰）
        """
        mec_names = ["mec0", "mec1", "mec2"]
        cpu_viol_before = sum(
            max(0.0, node_cpu_used[n] - self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        )
        projected = dict(node_cpu_used)  # copy
        for n in mec_names:
            cap = self.mec_nodes[n].cpu_capacity
            if projected[n] > cap:
                scale = cap / projected[n]
                projected[n] = cap
                # 記錄縮放比例（用於 t_comp 調整）
                self._last_cpu_scale[n] = scale
            else:
                self._last_cpu_scale[n] = 1.0
        return projected, cpu_viol_before

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
        ch_rem = self._channel_remaining()

        # 佇列變化率 Δqueue（V10 新增：飽和度預警）
        queue_delta = [
            float(queue_load_abs[i] - self._prev_queue[i])
            for i in range(3)
        ]

        # 最佳通道代理 SINR（以當前任務最大增益通道）
        current_gains = self._task_channel_gains[self.current_idx]
        best_sinr = float(np.max(current_gains))

        return np.array([
            float(task.data_size),
            float(task.deadline),
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],  # 3
            *mec_rem,                                               # 3
            *queue_load_abs,                                        # 3
            total_c,
            float(pressure),
            best_sinr,
            float(getattr(task, "task_type_id", 0)),
            *ch_rem,                                                # 3
            *queue_delta,                                           # 3 ← NEW
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────
    # Gym API
    # ────────────────────────────────────────────────────────────────

    def step(self, action):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]

        # ── 1. 部分卸載比例 ρ_u (公式 9，V10 新增) ────────────────
        rho = float(np.clip(action[15], 0.01, 1.0))  # 至少 1% 卸載

        # ── 2. NOMA 子通道指派（公式 2、3）────────────────────────
        # 使用 Gumbel softmax 採樣取代純 argmax，鼓勵通道多樣性
        channel_scores = action[12:15]
        ch_counts = self._channel_counts()

        # 先嘗試 Gumbel 採樣
        preferred_ch = self._gumbel_softmax_sample(
            channel_scores, self.channel_temp
        )

        channel_overflow = 0.0
        if ch_counts[preferred_ch] < MAX_NOMA_PER_CH:
            assigned_ch = preferred_ch
        else:
            # Fallback：找最空的通道
            assigned_ch = int(np.argmin(ch_counts))
            channel_overflow = 1.0

        self._channel_assignment[self.current_idx] = assigned_ch

        # ── 3. 計算 NOMA 速率 R_{u,k} (公式 10) ───────────────────
        sinr = self._compute_sinr(self.current_idx, assigned_ch)
        ru_k = self._sinr_to_rate(sinr)

        # 部分卸載上傳時間（公式 9）
        t_ul = rho * task.data_size / max(ru_k, 1e-6)

        # ── 4. VNF 放置 + CPU 分配（公式 11、12）──────────────────
        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios = action[9:12]

        node_cpu_used_raw = {n: 0.0 for n in mec_names}
        vnf_allocs = []       # [(sel_node, f_alloc, vnf.cpu_cycles)]
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
            node_cpu_used_raw[sel_node] += f_alloc
            vnf_allocs.append((sel_node, f_alloc, vnf.cpu_cycles))

        # ── 5. 可行化投影（公式 5、6，V10 新增）───────────────────
        node_cpu_used, cpu_viol = self._feasibility_projection(node_cpu_used_raw)

        # 根據縮放比例調整各 VNF 的實際分配算力 → 重算 t_comp
        t_comp = 0.0
        t_link = 0.0
        prev_node = None

        for i, (sel_node, f_alloc_orig, cpu_cycles) in enumerate(vnf_allocs):
            scale = self._last_cpu_scale[sel_node]
            f_alloc_proj = f_alloc_orig * scale
            f_alloc_proj = max(f_alloc_proj, 1e-6)

            t_comp += cpu_cycles / f_alloc_proj  # 公式 (11)

            if prev_node is not None and prev_node != sel_node:
                # 異質串接成本矩陣（V10，公式 12）
                t_link += BACKHAUL_DELAY.get((prev_node, sel_node), 2.0)

            prev_node = sel_node

        # ── 6. 端到端延遲（公式 13）───────────────────────────────
        delay = t_ul + t_comp + t_link

        # ── 7. Slack & 違規（公式 7、8）───────────────────────────
        slack = max(0.0, delay - task.deadline)
        deadline_pressure = delay / max(task.deadline, 1e-6)

        # ── 8. Reward（V10：channel_overflow 懲罰從 3.0 → 8.0）───
        cost = (
            1.0  * delay
            + self.beta * slack
            + 5.0  * cpu_viol
            + 0.5  * t_comp
            + 1.5  * deadline_pressure
            + 8.0  * channel_overflow     # ↑ 強化通道違規懲罰
        )
        reward = -cost / self.reward_scale

        # ── 9. 佇列更新 + 記錄前步佇列（V10，用於 Δqueue）────────
        self._prev_queue = [
            float(self.mec_nodes[n].queue_load) for n in mec_names
        ]
        for n in mec_names:
            self.mec_nodes[n].queue_load = (
                self.mec_nodes[n].queue_load * 0.65 + node_cpu_used[n] * 0.35
            )

        # ── 10. 下一步通道增益取樣（時變通道）────────────────────
        self._task_channel_gains[self.current_idx] = self._sample_channel_gains()

        self.current_idx += 1
        terminated = self.current_idx >= self.num_tasks

        obs = self._get_obs() if not terminated else np.zeros(21, dtype=np.float32)

        return obs, reward, terminated, False, {
            "delay":             delay,
            "slack":             slack,
            "cpu_viol":          cpu_viol,
            "t_ul":              t_ul,
            "t_comp":            t_comp,
            "t_link":            t_link,
            "deadline_pressure": deadline_pressure,
            "task_type_id":      getattr(task, "task_type_id", 0),
            "assigned_ch":       assigned_ch,
            "channel_overflow":  channel_overflow,
            "sinr":              sinr,
            "ru_k":              ru_k,
            "rho":               rho,
            "selected_nodes":    selected_nodes,
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

        # 通道增益矩陣 h_{u,k}
        self._task_channel_gains = np.array([
            self._sample_channel_gains() for _ in range(self.num_tasks)
        ])  # shape: (num_tasks, N_CHANNELS)

        # 子通道指派記錄，-1 表示尚未指派
        self._channel_assignment = [-1] * self.num_tasks

        # 算力縮放比例（可行化投影用）
        self._last_cpu_scale = {"mec0": 1.0, "mec1": 1.0, "mec2": 1.0}

        # 前一步佇列狀態（V10 Δqueue 特徵用）
        self._prev_queue = [0.0, 0.0, 0.0]

        self.current_idx = 0
        return self._get_obs(), {}
