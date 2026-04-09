import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

INPUT_FILE = "results/td3_v6_training_metrics.json"
OUTPUT_DIR = "results/figures_v6"


def smooth(data, window=15):
    """Simple moving average smoothing."""
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，請先執行 train_td3_v6.py")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    episodes = list(range(len(data["episode_rewards"])))

    # ------------------------------------------------------------------
    # 1. 個別指標圖 (4 張，V6 新增 cpu_viol)
    # ------------------------------------------------------------------
    metrics_cfg = [
        ("episode_rewards",      "Cumulative Reward",   "v6_reward.png",      "steelblue"),
        ("episode_avg_delay",    "Avg Delay (ms)",      "v6_delay.png",       "orangered"),
        ("episode_timeout_ratio","Timeout Ratio",       "v6_timeout.png",     "black"),
        ("episode_avg_cpu_viol", "Avg CPU Violation",   "v6_cpu_viol.png",    "purple"),
    ]

    for key, ylabel, fname, color in metrics_cfg:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(episodes, data[key], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(episodes, smooth(data[key]), color=color, linewidth=2.0, label="smoothed (w=15)")
        ax.set_title(f"V6 Training — {ylabel}", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"✅ 已存檔: {fname}")

    # ------------------------------------------------------------------
    # 2. 綜合面板圖 (2×2，方便論文插圖使用)
    # ------------------------------------------------------------------
    panel_keys = [
        ("episode_rewards",      "Cumulative Reward",   "steelblue"),
        ("episode_avg_delay",    "Avg Delay (ms)",      "orangered"),
        ("episode_timeout_ratio","Timeout Ratio",       "dimgray"),
        ("episode_avg_cpu_viol", "Avg CPU Violation",   "purple"),
    ]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    for idx, (key, ylabel, color) in enumerate(panel_keys):
        if key not in data:
            continue
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.plot(episodes, data[key], color="lightgray", alpha=0.35, linewidth=0.7)
        ax.plot(episodes, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.45)

    fig.suptitle("TD3 V6 Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    panel_path = os.path.join(OUTPUT_DIR, "v6_panel.png")
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已存檔: v6_panel.png")

    # ------------------------------------------------------------------
    # 3. 簡單統計摘要
    # ------------------------------------------------------------------
    print("\n📊 V6 訓練摘要 (最後 20 episodes 平均)：")
    for key, label, _ in panel_keys:
        if key in data and len(data[key]) >= 20:
            tail = data[key][-20:]
            print(f"  {label:25s}: {np.mean(tail):.4f}  (std {np.std(tail):.4f})")


if __name__ == "__main__":
    main()
