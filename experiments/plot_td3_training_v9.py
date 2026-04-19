import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUT_FILE = "results/td3_v9_training_metrics.json"
OUTPUT_DIR = "results/figures_v9"


def smooth(data, window=15):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


# ──────────────────────────────────────────────────────────────────────
# 圖表說明 (供論文參考)
#
# 1. Cumulative Reward       : 每 episode 的累計 reward，越接近 0 越好。
#                              快速收斂 → agent 找到有效策略。
#
# 2. Avg Delay (ms)          : 100 tasks 的平均端到端延遲 T_e2e (公式13)。
#                              論文核心指標，應持續下降並穩定。
#
# 3. Timeout Ratio           : 超時任務 (slack>0) 比例，對應論文 s_u 懲罰。
#                              理想值趨近 0。
#
# 4. Avg CPU Violation       : MEC 算力超用量均值 (公式5違規程度)，
#                              應趨近 0 代表資源可行解。
#
# 5. Deadline Pressure       : delay/deadline 比值。> 1 即超時；
#                              < 1 且越小表示餘裕越大。比 timeout ratio 更細緻。
#
# 6. Delay Decomposition     : t_ul / t_comp / t_link 疊加圖。
#                              直接看出端到端延遲的瓶頸在哪個環節。
#
# ── V9 新增 NOMA 指標 ────────────────────────────────────────────────
#
# 7. Avg SINR                : agent 指派通道後的平均 SINR (公式10分子/分母)。
#                              越高代表通道品質越好，t_ul 越短。
#                              若持續提升 → agent 學會避開高干擾通道。
#
# 8. Avg Channel Rate R_{u,k}: 平均有效上傳速率 (Mbps, 公式10)。
#                              與 SINR 正相關，直接決定 t_ul。
#                              提升代表 NOMA 通道管理改善。
#
# 9. Channel Overflow Ratio  : 子通道超用 (違反公式3 M上限) 的比例。
#                              應收斂至 0，代表 agent 學會遵守 NOMA 群組限制。
#
# 10. Channel Assignment Entropy : 子通道分配的 Shannon entropy (0~log2(3)≈1.58)。
#                              高值 → 均勻利用三個通道，避免單通道壅塞。
#                              低值 → agent 偏好固定通道，可能是次優解。
# ──────────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，請先執行 train_td3_v9.py")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps = list(range(len(data["episode_rewards"])))

    # ── 1. 個別指標圖 ─────────────────────────────────────────────────
    single_cfg = [
        ("episode_rewards",               "Cumulative Reward",        "v9_reward.png",           "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",           "v9_delay.png",            "orangered"),
        ("episode_timeout_ratio",         "Timeout Ratio",            "v9_timeout.png",          "black"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",        "v9_cpu_viol.png",         "purple"),
        ("episode_avg_deadline_pressure", "Deadline Pressure",        "v9_deadline_pressure.png","darkgreen"),
        # V9 NOMA 圖
        ("episode_avg_sinr",              "Avg SINR",                 "v9_sinr.png",             "#1A6FA8"),
        ("episode_avg_channel_rate",      "Avg Channel Rate R (Mbps)","v9_channel_rate.png",     "#0F6E56"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio",   "v9_ch_overflow.png",      "#D85A30"),
        ("episode_avg_channel_entropy",   "Channel Assignment Entropy","v9_ch_entropy.png",      "#7F77DD"),
    ]

    for key, ylabel, fname, color in single_cfg:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, data[key], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0, label="smoothed (w=15)")
        ax.set_title(f"V9 Training — {ylabel}", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close(fig)
        print(f"✅ {fname}")

    # ── 2. 延遲分解圖 ────────────────────────────────────────────────
    decomp_keys   = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
    decomp_labels = ["Upload t_ul (NOMA)", "Compute t_comp", "Link t_link"]
    decomp_colors = ["#4C9BE8", "#E87B4C", "#6DBF67"]

    if all(k in data for k in decomp_keys):
        fig, ax = plt.subplots(figsize=(12, 5))
        bottoms = np.zeros(len(eps))
        for key, label, color in zip(decomp_keys, decomp_labels, decomp_colors):
            vals = np.array(smooth(data[key]))
            ax.fill_between(eps, bottoms, bottoms + vals,
                            alpha=0.75, color=color, label=label)
            bottoms += vals
        ax.set_title("V9 Training — Delay Decomposition (smoothed)", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Delay (ms)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "v9_delay_decomp.png"), dpi=300)
        plt.close(fig)
        print("✅ v9_delay_decomp.png")

    # ── 3. V9 NOMA 四合一面板 ─────────────────────────────────────────
    noma_cfg = [
        ("episode_avg_sinr",              "Avg SINR",                "#1A6FA8"),
        ("episode_avg_channel_rate",      "Avg Channel Rate (Mbps)", "#0F6E56"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio",  "#D85A30"),
        ("episode_avg_channel_entropy",   "Channel Entropy",         "#7F77DD"),
    ]
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)
    for idx, (key, ylabel, color) in enumerate(noma_cfg):
        if key not in data: continue
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.plot(eps, data[key], color="lightgray", alpha=0.3, linewidth=0.7)
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("TD3 V9 — NOMA Channel Metrics", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.savefig(os.path.join(OUTPUT_DIR, "v9_noma_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ v9_noma_panel.png")

    # ── 4. 主要指標六合一面板 ─────────────────────────────────────────
    panel_cfg = [
        ("episode_rewards",               "Cumulative Reward",  "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",     "orangered"),
        ("episode_timeout_ratio",         "Timeout Ratio",      "dimgray"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",  "purple"),
        ("episode_avg_t_comp",            "Avg t_comp (ms)",    "#E87B4C"),
        ("episode_avg_channel_rate",      "Avg Channel Rate",   "#0F6E56"),
    ]
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    for idx, (key, ylabel, color) in enumerate(panel_cfg):
        if key not in data: continue
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.plot(eps, data[key], color="lightgray", alpha=0.3, linewidth=0.7)
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("TD3 V9 Training Metrics", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v9_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ v9_panel.png")

    # ── 5. 數值摘要 ────────────────────────────────────────────────────
    print("\n📊 V9 訓練摘要 (最後 20 episodes 平均)：")
    report = [
        ("episode_avg_delay",             "Avg Delay             "),
        ("episode_timeout_ratio",         "Timeout Ratio         "),
        ("episode_avg_cpu_viol",          "Avg CPU Viol          "),
        ("episode_avg_deadline_pressure", "Deadline Pressure     "),
        ("episode_avg_t_ul",              "Avg t_ul              "),
        ("episode_avg_t_comp",            "Avg t_comp            "),
        ("episode_avg_t_link",            "Avg t_link            "),
        ("episode_avg_sinr",              "Avg SINR              "),
        ("episode_avg_channel_rate",      "Avg Channel Rate      "),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio"),
        ("episode_avg_channel_entropy",   "Channel Entropy       "),
        ("episode_rewards",               "Avg Reward            "),
    ]
    for key, label in report:
        if key in data and len(data[key]) >= 20:
            tail = data[key][-20:]
            print(f"  {label}: {np.mean(tail):.4f}  ± {np.std(tail):.4f}")


if __name__ == "__main__":
    main()
