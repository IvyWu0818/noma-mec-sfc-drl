import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUT_FILE = "results/td3_v10_training_metrics.json"
OUTPUT_DIR = "results/figures_v10"


def smooth(data, window=15):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


# ──────────────────────────────────────────────────────────────────────
# 圖表說明 (供論文參考)
#
# V10 相較 V9 的主要改進（可在圖表中觀察）：
#
# 1. Channel Overflow Ratio  ← 最關鍵
#    V9 穩定在 0.95~0.97（幾乎全部違規）
#    V10 目標：收斂至 0.3 以下，代表 Gumbel softmax + 強化懲罰有效
#
# 2. Avg CPU Violation
#    V10 加入可行化投影後，cpu_viol 應趨近 0
#    這才符合論文公式(5)的硬約束要求
#
# 3. Avg Rho (部分卸載比例)
#    新增指標。agent 會學習在通道壅塞時降低 rho（減少上傳量）
#    以換取更低的 t_ul。理想值在 0.6~0.9 之間穩定。
#
# 4. Avg t_link (異質串接延遲)
#    V10 使用異質 backhaul 矩陣（非固定 2ms）
#    若 agent 學會將相鄰 VNF 放在同節點，t_link 應下降
#
# 5. Channel Assignment Entropy
#    V10 用 Gumbel softmax 鼓勵通道多樣性
#    entropy 應維持在 1.4~1.58（接近 log2(3)=1.58）
#    V9 因為固定在 1.585 代表完全均勻（暗示 fallback 主導，不是學習）
# ──────────────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，請先執行 train_td3_v10.py")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps = list(range(len(data["episode_rewards"])))

    # ── 1. 個別指標圖 ─────────────────────────────────────────────────
    single_cfg = [
        ("episode_rewards",               "Cumulative Reward",         "v10_reward.png",            "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",            "v10_delay.png",             "orangered"),
        ("episode_timeout_ratio",         "Timeout Ratio",             "v10_timeout.png",           "black"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",         "v10_cpu_viol.png",          "purple"),
        ("episode_avg_deadline_pressure", "Deadline Pressure",         "v10_deadline_pressure.png", "darkgreen"),
        # V9 繼承 NOMA 圖
        ("episode_avg_sinr",              "Avg SINR",                  "v10_sinr.png",              "#1A6FA8"),
        ("episode_avg_channel_rate",      "Avg Channel Rate R (Mbps)", "v10_channel_rate.png",      "#0F6E56"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio",    "v10_ch_overflow.png",       "#D85A30"),
        ("episode_avg_channel_entropy",   "Channel Assignment Entropy","v10_ch_entropy.png",        "#7F77DD"),
        # V10 新增圖
        ("episode_avg_rho",               "Avg Offload Ratio ρ",       "v10_rho.png",               "#B8860B"),
        ("episode_avg_queue_delta",       "Avg Queue Delta |Δq|",      "v10_queue_delta.png",       "#8B4513"),
    ]

    for key, ylabel, fname, color in single_cfg:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, data[key], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0, label="smoothed (w=15)")
        ax.set_title(f"V10 Training — {ylabel}", fontsize=13)
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
    decomp_labels = ["Upload t_ul (NOMA × ρ)", "Compute t_comp", "Link t_link (異質)"]
    decomp_colors = ["#4C9BE8", "#E87B4C", "#6DBF67"]

    if all(k in data for k in decomp_keys):
        fig, ax = plt.subplots(figsize=(12, 5))
        bottoms = np.zeros(len(eps))
        for key, label, color in zip(decomp_keys, decomp_labels, decomp_colors):
            vals = np.array(smooth(data[key]))
            ax.fill_between(eps, bottoms, bottoms + vals,
                            alpha=0.75, color=color, label=label)
            bottoms += vals
        ax.set_title("V10 Training — Delay Decomposition (smoothed)", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Delay (ms)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "v10_delay_decomp.png"), dpi=300)
        plt.close(fig)
        print("✅ v10_delay_decomp.png")

    # ── 3. V10 NOMA 四合一面板 ────────────────────────────────────────
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
    fig.suptitle("TD3 V10 — NOMA Channel Metrics", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.savefig(os.path.join(OUTPUT_DIR, "v10_noma_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ v10_noma_panel.png")

    # ── 4. V10 新增指標雙合一面板 ─────────────────────────────────────
    v10_new_cfg = [
        ("episode_avg_rho",         "Avg Offload Ratio ρ",   "#B8860B"),
        ("episode_avg_queue_delta", "Avg Queue Delta |Δq|",  "#8B4513"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (key, ylabel, color) in zip(axes, v10_new_cfg):
        if key not in data: continue
        ax.plot(eps, data[key], color="lightgray", alpha=0.3, linewidth=0.7)
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("TD3 V10 — New Metrics (Partial Offload + Queue Delta)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "v10_new_metrics.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ v10_new_metrics.png")

    # ── 5. 主要指標六合一面板 ─────────────────────────────────────────
    panel_cfg = [
        ("episode_rewards",               "Cumulative Reward",  "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",     "orangered"),
        ("episode_timeout_ratio",         "Timeout Ratio",      "dimgray"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",  "purple"),
        ("episode_avg_rho",               "Avg Offload Ratio ρ","#B8860B"),
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
    fig.suptitle("TD3 V10 Training Metrics", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v10_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ v10_panel.png")

    # ── 6. V9 vs V10 比較圖（若同時有兩份數據）──────────────────────
    v9_file = "results/td3_v9_training_metrics.json"
    if os.path.exists(v9_file):
        with open(v9_file, "r") as f:
            v9_data = json.load(f)

        compare_keys = [
            ("episode_rewards",            "Cumulative Reward"),
            ("episode_avg_delay",          "Avg Delay (ms)"),
            ("episode_channel_overflow_ratio", "Channel Overflow Ratio"),
            ("episode_avg_cpu_viol",       "Avg CPU Violation"),
        ]
        fig = plt.figure(figsize=(16, 8))
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
        for idx, (key, ylabel) in enumerate(compare_keys):
            if key not in data or key not in v9_data: continue
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            eps_v9 = list(range(len(v9_data[key])))
            ax.plot(eps_v9, smooth(v9_data[key]), color="gray",
                    linewidth=1.8, linestyle="--", label="V9")
            ax.plot(eps, smooth(data[key]), color="steelblue",
                    linewidth=2.0, label="V10")
            ax.set_title(ylabel, fontsize=11)
            ax.set_xlabel("Episode", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
        fig.suptitle("V9 vs V10 Comparison", fontsize=14,
                     fontweight="bold", y=1.01)
        fig.savefig(os.path.join(OUTPUT_DIR, "v9_vs_v10_compare.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("✅ v9_vs_v10_compare.png")

    # ── 7. 數值摘要 ────────────────────────────────────────────────────
    print("\n📊 V10 訓練摘要 (最後 20 episodes 平均)：")
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
        ("episode_avg_rho",               "Avg Rho (卸載比例)    "),
        ("episode_avg_queue_delta",       "Avg Queue Delta       "),
        ("episode_rewards",               "Avg Reward            "),
    ]
    for key, label in report:
        if key in data and len(data[key]) >= 20:
            tail = data[key][-20:]
            print(f"  {label}: {np.mean(tail):.4f}  ± {np.std(tail):.4f}")


if __name__ == "__main__":
    main()
