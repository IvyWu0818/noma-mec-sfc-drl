import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUT_FILE = "results/td3_v7_training_metrics.json"
OUTPUT_DIR = "results/figures_v7"


def smooth(data, window=15):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，請先執行 train_td3_v7.py")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps = list(range(len(data["episode_rewards"])))

    # ------------------------------------------------------------------
    # 1. 個別指標圖
    # ------------------------------------------------------------------
    single_cfg = [
        ("episode_rewards",      "Cumulative Reward",  "v7_reward.png",   "steelblue"),
        ("episode_avg_delay",    "Avg Delay (ms)",     "v7_delay.png",    "orangered"),
        ("episode_timeout_ratio","Timeout Ratio",      "v7_timeout.png",  "black"),
        ("episode_avg_cpu_viol", "Avg CPU Violation",  "v7_cpu_viol.png", "purple"),
    ]
    for key, ylabel, fname, color in single_cfg:
        if key not in data: continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, data[key], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0, label="smoothed (w=15)")
        ax.set_title(f"V7 Training — {ylabel}", fontsize=13)
        ax.set_xlabel("Episode"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=10); ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close(fig)
        print(f"✅ {fname}")

    # ------------------------------------------------------------------
    # 2. 延遲分解圖
    # ------------------------------------------------------------------
    decomp_keys   = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
    decomp_labels = ["Upload (t_ul)", "Compute (t_comp)", "Link (t_link)"]
    decomp_colors = ["#4C9BE8", "#E87B4C", "#6DBF67"]

    if all(k in data for k in decomp_keys):
        fig, ax = plt.subplots(figsize=(12, 5))
        bottoms = np.zeros(len(eps))
        for key, label, color in zip(decomp_keys, decomp_labels, decomp_colors):
            vals = np.array(smooth(data[key]))
            ax.fill_between(eps, bottoms, bottoms + vals, alpha=0.75, color=color, label=label)
            bottoms += vals
        ax.set_title("V7 Training — Delay Decomposition (smoothed)", fontsize=13)
        ax.set_xlabel("Episode"); ax.set_ylabel("Avg Delay (ms)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "v7_delay_decomp.png"), dpi=300)
        plt.close(fig)
        print("✅ v7_delay_decomp.png")

    # ------------------------------------------------------------------
    # 3. 綜合面板 (2×3)
    # ------------------------------------------------------------------
    panel_cfg = [
        ("episode_rewards",      "Cumulative Reward",  "steelblue"),
        ("episode_avg_delay",    "Avg Delay (ms)",     "orangered"),
        ("episode_timeout_ratio","Timeout Ratio",      "dimgray"),
        ("episode_avg_cpu_viol", "Avg CPU Violation",  "purple"),
        ("episode_avg_t_comp",   "Avg t_comp (ms)",    "#E87B4C"),
        ("episode_avg_t_ul",     "Avg t_ul (ms)",      "#4C9BE8"),
    ]
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    for idx, (key, ylabel, color) in enumerate(panel_cfg):
        if key not in data: continue
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.plot(eps, data[key], color="lightgray", alpha=0.3, linewidth=0.7)
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("TD3 V7 Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v7_panel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ v7_panel.png")

    # ------------------------------------------------------------------
    # 4. 數值摘要
    # ------------------------------------------------------------------
    print("\n📊 V7 訓練摘要 (最後 20 episodes 平均)：")
    report = [
        ("episode_avg_delay",    "Avg Delay      "),
        ("episode_timeout_ratio","Timeout Ratio  "),
        ("episode_avg_cpu_viol", "Avg CPU Viol   "),
        ("episode_avg_t_ul",     "Avg t_ul       "),
        ("episode_avg_t_comp",   "Avg t_comp     "),
        ("episode_avg_t_link",   "Avg t_link     "),
        ("episode_rewards",      "Avg Reward     "),
    ]
    for key, label in report:
        if key in data and len(data[key]) >= 20:
            tail = data[key][-20:]
            print(f"  {label}: {np.mean(tail):.4f}  ± {np.std(tail):.4f}")


if __name__ == "__main__":
    main()
