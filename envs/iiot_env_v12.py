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

# ── 子通道配置 ────────────────────────────────────────────────────────────
N_CHANNELS      = 3     # 子通道數 K = 3
                        # 總系統頻寬 = K × B_k = 3 × 9 = 27 MHz
                        # 對應 5G NR FR1 sub-6GHz 一個 Component Carrier (~25-30 MHz)

MAX_NOMA_PER_CH = 2     # 每通道最多同時用戶數 M = 2
                        # 依據: SIC 接收器複雜度為 O(M!)，M=2 時僅需 2 步解碼
                        # M > 2 的誤差傳播 (error propagation) 顯著影響 BER
                        # 3GPP TR 38.812 Section 6.1.2: 研究重點為 M ≤ 2 場景
                        # 本模型採用 M=2 符合 5G NOMA 主流研究設定

# ── 無線通道參數 ─────────────────────────────────────────────────────────
BANDWIDTH_MHZ   = 9.0   # 每通道頻寬 B_k = 9 MHz
                        # Shannon 容量: R [Mbps] = B [MHz] × log₂(1 + SINR)
                        # 等效: R [bps] = B [Hz] × log₂(1 + SINR)
                        # 傳輸延遲: t_ul [ms] = data_size [Kbits] / R [Mbps]
                        #   推導: 1 Kbit / 1 Mbps = 10³ bit / 10⁶ bps = 10⁻³ s = 1 ms ✓

TX_POWER_W      = 0.2   # UE 發射功率 P_u = 0.2 W = 200 mW = 23 dBm
                        # 依據: 3GPP TS 38.101-1 Table 6.2.1-1, Power Class 3
                        # 適用於一般 5G NR sub-6GHz 手機/IIoT 裝置

NOISE_POWER_W   = 4e-3  # 接收端噪聲功率 N = 4 mW（正規化值）
                        # 物理參考: kTFB = 1.38e-23 × 290 × 10 × 9e6 ≈ 3.6e-13 W
                        # 此處採正規化 (normalized) 模型，通道增益 |h|² 已吸收大尺度
                        # 路徑損耗 (path loss) 與陰影衰落 (shadowing)
                        # 正規化後 SNR = P × E[|h|²] / N = 0.2 × 1 / 4e-3 = 50 ≈ 17 dB
                        # 對應 5G NR 中距離 (mid-range) 典型 SNR 15-20 dB ✓

# ── 時間槽配置 ────────────────────────────────────────────────────────────
TIME_SLOT_MS    = 0.5   # 5G NR 時槽長度 = 0.5 ms
                        # 依據: numerology µ=1 (30 kHz SCS), 1 slot = 14 symbols = 0.5 ms
                        # 3GPP TS 38.211 Table 4.3.2-1

# ── 截止時間依據 ─────────────────────────────────────────────────────────
# Task deadline (T_max) = 10–22 ms，對應 5G IIoT 製程自動化場景
# 依據: 3GPP TR 22.804 Table 5.1 "Communication service availability"
#   製程自動化 (Process Automation):    T_max = 50 ms
#   閉迴路運動控制 (Motion Control):    T_max = 1 ms
#   電力配電自動化 (Power Distribution): T_max = 3 ms
# 本模型設定 10–22 ms：比製程自動化更嚴苛，比運動控制寬鬆，
# 代表「中等即時性」IIoT 應用（感測融合、邊緣推論、AGV 協調等）
#
# 如需 URLLC 場景，應將 T_max 降至 1–5 ms 並相應調整算力/頻寬
# 100 ms 設定過於寬鬆（等同 eMBB），無法有效測試 MEC 卸載優化

# ── MEC 算力單位說明 ──────────────────────────────────────────────────────
# cpu_capacity 單位: MCycles/ms (= GHz)，即每毫秒可執行的百萬浮點指令數
#   mec0 = 35 GHz, mec1 = 45 GHz, mec2 = 55 GHz
#   代表不同等級的 MEC 節點（低/中/高性能邊緣伺服器）
#   典型 5G MEC server: Intel Xeon 30-100 GHz 等效算力
#
# VNF cpu_cycles 單位: MCycles（百萬指令數）
#   計算延遲: t_comp [ms] = cpu_cycles [MCycles] / f_alloc [MCycles/ms] ✓

# ── 背程延遲矩陣 ─────────────────────────────────────────────────────────
BACKHAUL_DELAY_MS = {   # 單位: ms，MEC 節點間背程 (backhaul) 傳輸延遲
    ("mec0", "mec0"): 0.0,
    ("mec0", "mec1"): 1.5,
    ("mec0", "mec2"): 3.0,
    ("mec1", "mec0"): 1.5,
    ("mec1", "mec1"): 0.0,
    ("mec1", "mec2"): 2.0,
    ("mec2", "mec0"): 3.0,
    ("mec2", "mec1"): 2.0,
    ("mec2", "mec2"): 0.0,
    # 依據: 5G Xhaul 典型延遲，光纖 fronthaul < 1 ms，
    # wireless backhaul 1–5 ms (3GPP TR 38.801 Section 9.1)
}


class IIoTEnvV12(gym.Env):
    """
    V12: 對齊 5G NR 與 NOMA 標準的參數修正版本

    ═══════════════════════════════════════════════════════════════════
    V11 → V12 的核心改變（5G/NOMA 合規性修正）
    ═══════════════════════════════════════════════════════════════════

    [修正 1] TX_POWER: 0.5 W → 0.2 W (23 dBm)
      依據: 3GPP TS 38.101-1 Power Class 3
      0.5 W (27 dBm) 超過 5G NR UE 規範上限

    [修正 2] NOISE_POWER: 0.01 → 4e-3 (與 TX_POWER 同比例調整)
      維持 SNR ≈ 50 (17 dB)，對應 5G NR 中距離場景

    [修正 3] TIME_SLOT_MS: 5.0 ms → 0.5 ms
      對齊 5G NR numerology µ=1 (30 kHz SCS) 時槽定義
      (3GPP TS 38.211 Table 4.3.2-1)

    [修正 4] 通道增益 |h|² 標記修正
      h_{u,k} ~ Rayleigh → |h_{u,k}|² ~ Exp(1) (功率增益)
      SINR 公式使用功率增益 |h|²，非振幅 h
      避免「h 為虛數」的混淆：|h|² 永遠為實數正值

    [修正 5] 觀測中的 SINR 修正
      V11: best_sinr = max(channel_gains)  ← 回傳的是 |h|² 不是 SINR
      V12: best_sinr = P × max(|h|²) / N  ← 真實單用戶 SINR（對數尺度）
      再取 log2(1 + SINR) 轉為歸一化頻譜效率 [bps/Hz]，供 agent 評估通道品質

    [修正 6] 全面補充單位標注
      data_size: Kbits | deadline: ms | cpu_cycles: MCycles
      cpu_capacity: MCycles/ms (=GHz) | bandwidth: MHz | power: W

    [修正 7] M=2 與 T_max 依據說明
      見頂部常數區的詳細 5G 標準引用

    ═══════════════════════════════════════════════════════════════════
    Obs 維度 (21)：同 V11，但 sinr obs 改為正規化頻譜效率 log₂(1+SINR)
    Action 維度 (16)：同 V11
    ═══════════════════════════════════════════════════════════════════

    傳輸延遲單位驗證:
      t_ul [ms] = data_size [Kbits] / (B [MHz] × log₂(1+SINR) [bps/Hz])
                = data_size [Kbits] / R [Mbps]
                = (D × 10³ bits) / (R × 10⁶ bps) × 10³ ms/s
                = D / R  [ms]  ✓

    計算延遲單位驗證:
      t_comp [ms] = cpu_cycles [MCycles] / f_alloc [MCycles/ms]  ✓
    """

    SLOT_TASK_SIZE = N_CHANNELS * MAX_NOMA_PER_CH  # = 6 tasks per time-slot

    def __init__(self, num_tasks=100, beta=12.0, seed=42, reward_scale=50.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.beta = beta
        self.reward_scale = reward_scale
        self.np_random = np.random.default_rng(seed)

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
        """
        瑞利衰落功率增益 |h_{u,k}|² ~ Exp(1)，clip [0.1, 2.0]

        說明: 在 Rayleigh fading 中，複數通道增益 h = h_r + j·h_i，
        其中 h_r, h_i ~ N(0, 0.5)，使得 |h|² = h_r² + h_i² ~ Exp(1)。
        SINR 公式使用功率增益 |h|²（實數正值），不直接使用複數 h，
        因此不存在虛數問題。變數名 `h_sq` 表示 |h|²。
        """
        h_sq = self.np_random.exponential(scale=1.0, size=N_CHANNELS).astype(np.float32)
        return np.clip(h_sq, 0.1, 2.0)

    def _compute_sinr(self, task_idx: int, ch: int) -> float:
        """
        上行 NOMA SINR（公式 10）。

        SIC 解碼順序: 先解碼 |h|² 較大（信號較強）的用戶。
        對用戶 u，干擾來自同 slot 同通道中 |h|² < |h_u|² 的其他用戶
        （這些較弱用戶尚未被消除，是殘留干擾）。

        SINR_u = P_u × |h_u|² / ( Σ_{j: |h_j|² < |h_u|²} P_j × |h_j|² + N )
        """
        h_sq_u = float(self._task_channel_gains[task_idx][ch])  # |h_u|²
        slot_start = (self.current_idx // self.SLOT_TASK_SIZE) * self.SLOT_TASK_SIZE
        interference = 0.0
        for other_idx in range(slot_start, self.current_idx):
            if self._channel_assignment[other_idx] == ch:
                h_sq_other = float(self._task_channel_gains[other_idx][ch])
                if h_sq_other < h_sq_u:  # 弱用戶造成干擾
                    interference += TX_POWER_W * h_sq_other
        sinr = (TX_POWER_W * h_sq_u) / (interference + NOISE_POWER_W)
        return float(sinr)

    def _sinr_to_rate(self, sinr: float) -> float:
        """
        Shannon 容量公式 (公式 10): R_{u,k} [Mbps] = B_k [MHz] × log₂(1 + SINR)

        單位說明:
          B_k = 9 MHz, SINR = P|h|²/N (無量綱)
          R = 9 × log₂(1 + 50) ≈ 51 Mbps (典型單用戶，無干擾)
          t_ul [ms] = data_size [Kbits] / R [Mbps] ✓
        """
        return BANDWIDTH_MHZ * np.log2(1.0 + sinr)  # Mbps

    def _slot_ch_remaining(self) -> list:
        """當前 time-slot 各通道剩餘容量比例（公式 3 約束的真實狀態）"""
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
        if p < 0.35:   return 0   # urgent (緊急型)
        elif p < 0.70: return 1   # compute-heavy (計算密集型)
        else:          return 2   # bandwidth-heavy (頻寬密集型)

    def _build_task_by_type(self, task_id: int, task_type: int) -> Task:
        """
        任務參數採樣。單位:
          data_size : Kbits  (千位元)
          deadline  : ms     (毫秒), 對應 5G IIoT 中等即時性應用 10–22 ms
          cpu_cycles: MCycles (百萬指令數)
        """
        rng = random.Random(int(self.np_random.integers(0, 2**31)))
        if task_type == 0:   # 緊急型: 小資料、短 deadline、輕計算
            data_size = rng.randint(18, 32)   # Kbits
            deadline  = rng.randint(10, 16)   # ms
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]  # MCycles
        elif task_type == 1: # 計算密集型: 中資料、中 deadline、重計算
            data_size = rng.randint(25, 40)   # Kbits
            deadline  = rng.randint(14, 22)   # ms
            vnfs = [VNF(j, rng.randint(18, 30)) for j in range(3)]  # MCycles
        else:                # 頻寬密集型: 大資料、中 deadline、輕計算
            data_size = rng.randint(40, 65)   # Kbits
            deadline  = rng.randint(14, 22)   # ms
            vnfs = [VNF(j, rng.randint(10, 18)) for j in range(3)]  # MCycles
        task = Task(task_id, data_size, deadline, SFC(vnfs))
        task.task_type_id = task_type
        return task

    def _get_obs(self):
        task = self.tasks[self.current_idx]
        mec_names = ["mec0", "mec1", "mec2"]
        total_c = float(sum(v.cpu_cycles for v in task.sfc_chain.vnfs))  # MCycles

        mec_rem = [
            max(0.0,
                (self.mec_nodes[n].cpu_capacity - self.mec_nodes[n].queue_load)
                / self.mec_nodes[n].cpu_capacity)
            for n in mec_names
        ]
        queue_load_abs = [float(self.mec_nodes[n].queue_load) for n in mec_names]
        pressure = (task.data_size + total_c) / max(task.deadline, 1)

        ch_rem = self._slot_ch_remaining()

        queue_delta = [
            float(queue_load_abs[i] - self._prev_queue[i])
            for i in range(3)
        ]

        # V12 修正: 回傳正規化頻譜效率 log₂(1 + SINR_best)，而非 max(|h|²)
        # SINR_best = 單用戶（無干擾）在最佳通道的 SINR = P × max(|h|²) / N
        # log₂(1 + SINR) 單位: bps/Hz，數值範圍典型 3–7 bps/Hz，對 agent 更直觀
        best_h_sq = float(np.max(self._task_channel_gains[self.current_idx]))
        sinr_best = TX_POWER_W * best_h_sq / NOISE_POWER_W
        spectral_eff = float(np.log2(1.0 + sinr_best))  # bps/Hz (歸一化頻譜效率)

        return np.array([
            float(task.data_size),          # Kbits
            float(task.deadline),           # ms
            *[float(v.cpu_cycles) for v in task.sfc_chain.vnfs],  # MCycles ×3
            *mec_rem,                       # 各 MEC 算力剩餘比例 [0,1] ×3
            *queue_load_abs,                # 各 MEC 佇列負載 MCycles/ms ×3
            total_c,                        # 任務總計算量 MCycles
            float(pressure),                # 壓力指標 (data+cpu)/deadline
            spectral_eff,                   # 最佳通道歸一化頻譜效率 bps/Hz (V12 修正)
            float(getattr(task, "task_type_id", 0)),
            *ch_rem,                        # per-slot 通道剩餘容量 [0,1] ×3
            *queue_delta,                   # 佇列變化量 ×3
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
        channel_scores = action[12:15]
        preferred_ch   = int(np.argmax(channel_scores))

        channel_overflow = 0.0
        if self._slot_ch_count[preferred_ch] < MAX_NOMA_PER_CH:
            assigned_ch = preferred_ch
        else:
            assigned_ch = int(np.argmin(self._slot_ch_count))
            channel_overflow = 1.0

        self._channel_assignment[self.current_idx] = assigned_ch
        self._slot_ch_count[assigned_ch] += 1

        # ── 3. NOMA 上行速率 R_{u,k} (公式 10) ────────────────────
        # SINR = P_u × |h_{u,k}|² / (干擾 + N)  [無量綱]
        # R    = B_k × log₂(1 + SINR)            [Mbps]
        # t_ul = ρ × data_size / R               [ms]  (data_size 單位: Kbits)
        sinr = self._compute_sinr(self.current_idx, assigned_ch)
        ru_k = self._sinr_to_rate(sinr)                         # Mbps
        t_ul = rho * task.data_size / max(ru_k, 1e-6)          # ms

        # ── 4. VNF 放置 + CPU 分配 ─────────────────────────────────
        placement_scores = action[:9].reshape(3, 3)
        cpu_ratios       = action[9:12]

        node_cpu_used_raw = {n: 0.0 for n in mec_names}
        vnf_allocs, selected_nodes = [], []

        for i, vnf in enumerate(task.sfc_chain.vnfs):
            c_idx    = int(np.argmax(placement_scores[i]))
            sel_node = mec_names[c_idx]
            selected_nodes.append(sel_node)

            node_cap = self.mec_nodes[sel_node].cpu_capacity  # MCycles/ms
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
            t_comp += cpu_cycles / f_proj                        # ms = MCycles / (MCycles/ms)
            if prev_node is not None and prev_node != sel_node:
                t_link += BACKHAUL_DELAY_MS.get((prev_node, sel_node), 2.0)  # ms
            prev_node = sel_node

        # ── 6. 端到端延遲（公式 13）───────────────────────────────
        delay = t_ul + t_comp + t_link                          # ms

        # ── 7. Slack & 違規 ────────────────────────────────────────
        slack             = max(0.0, delay - task.deadline)     # ms
        deadline_pressure = delay / max(task.deadline, 1e-6)

        # ── 8. Reward ──────────────────────────────────────────────
        cost = (
            1.0        * delay
            + self.beta * slack
            + 5.0       * cpu_viol
            + 0.5       * t_comp
            + 1.5       * deadline_pressure
            + 5.0       * channel_overflow
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

        # ── 10. Time-slot 切換 ─────────────────────────────────────
        self.current_idx += 1
        if self.current_idx % self.SLOT_TASK_SIZE == 0:
            self._slot_ch_count = [0] * N_CHANNELS

        # ── 11. 時變通道增益更新 ───────────────────────────────────
        prev_idx = self.current_idx - 1
        self._task_channel_gains[prev_idx] = self._sample_channel_gains()

        terminated = self.current_idx >= self.num_tasks
        obs = self._get_obs() if not terminated else np.zeros(21, dtype=np.float32)

        return obs, float(reward), terminated, False, {
            "delay":             float(delay),       # ms
            "slack":             float(slack),       # ms
            "cpu_viol":          float(cpu_viol),    # MCycles/ms
            "t_ul":              float(t_ul),        # ms
            "t_comp":            float(t_comp),      # ms
            "t_link":            float(t_link),      # ms
            "deadline_pressure": float(deadline_pressure),
            "task_type_id":      int(getattr(task, "task_type_id", 0)),
            "assigned_ch":       int(assigned_ch),
            "channel_overflow":  float(channel_overflow),
            "sinr":              float(sinr),        # 無量綱 (linear scale)
            "sinr_db":           float(10 * np.log10(max(sinr, 1e-10))),  # dB
            "ru_k":              float(ru_k),        # Mbps
            "rho":               float(rho),
            "selected_nodes":    selected_nodes,
            "slot_id":           int((self.current_idx - 1) // self.SLOT_TASK_SIZE),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # MEC 節點算力單位: MCycles/ms (= GHz)
        # mec0=35 GHz (低性能邊緣節點), mec1=45 GHz (中), mec2=55 GHz (高)
        self.mec_nodes = {
            "mec0": MECNode("mec0", 35),   # 35 MCycles/ms = 35 GHz
            "mec1": MECNode("mec1", 45),   # 45 MCycles/ms = 45 GHz
            "mec2": MECNode("mec2", 55),   # 55 MCycles/ms = 55 GHz
        }

        self.tasks = []
        for i in range(self.num_tasks):
            task_type = self._sample_task_regime()
            self.tasks.append(self._build_task_by_type(i, task_type))

        # 各任務在各通道的功率增益 |h_{u,k}|²（實數正值）
        self._task_channel_gains = np.array([
            self._sample_channel_gains() for _ in range(self.num_tasks)
        ])

        self._channel_assignment = [-1] * self.num_tasks
        self._slot_ch_count      = [0] * N_CHANNELS
        self._last_cpu_scale     = {"mec0": 1.0, "mec1": 1.0, "mec2": 1.0}
        self._prev_queue         = [0.0, 0.0, 0.0]
        self.current_idx         = 0
        return self._get_obs(), {}
