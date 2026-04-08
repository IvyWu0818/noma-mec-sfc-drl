import os
import json
import matplotlib.pyplot as plt

# ============================================================
# 設定：將輸出路徑改為 v4，與 V3 徹底分開
# ============================================================
INPUT_FILE = "results/td3_v4_training_metrics.json"
OUTPUT_DIR = "results/figures_v4"
SHOW_PLOTS = False  

# ============================================================
# 工具：平滑函式 (讓趨勢更清晰)
# ============================================================
def smooth(data, window=10):
    if not data:
        return []
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(sum(data[start:i + 1]) / (i - start + 1))
    return smoothed

# ============================================================
# 存圖函式
# ============================================================
def save_plot(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ 圖片已儲存至: {path}")

# ============================================================
# 主程式
# ============================================================
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到資料檔: {INPUT_FILE}，請先執行訓練。")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 讀取資料
    rewards = data.get("episode_rewards", [])
    delays = data.get("episode_avg_delay", [])
    slacks = data.get("episode_avg_slack", [])
    timeouts = data.get("episode_timeout_ratio", [])
    actor_losses = data.get("actor_losses", [])
    critic_losses = data.get("critic_losses", [])

    # 設定全域字體風格 (可選)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    # 1. Episode Reward (對齊論文目標：最大化 Reward)
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='lightgray', alpha=0.5, label="Raw")
    plt.plot(smooth(rewards, window=20), color='blue', linewidth=2, label="Smoothed")
    plt.title("V4 Training: Total Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig1, "v4_reward.png")

    # 2. Average Delay (論文公式 13: 端到端延遲趨勢)
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(delays, color='peachpuff', alpha=0.6, label="Raw")
    plt.plot(smooth(delays, window=20), color='orangered', linewidth=2, label="Smoothed")
    plt.title("V4 Training: End-to-End Delay (ms)")
    plt.xlabel("Episode")
    plt.ylabel("Delay (ms)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig2, "v4_delay.png")

    # 3. Timeout Ratio (論文關鍵指標：任務超時風險)
    fig3 = plt.figure(figsize=(10, 6))
    plt.plot(timeouts, color='silver', alpha=0.6, label="Raw")
    plt.plot(smooth(timeouts, window=20), color='black', linewidth=2, label="Smoothed")
    plt.title("V4 Training: Timeout Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Ratio (0.0 - 1.0)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(fig3, "v4_timeout.png")

    # 4. Losses (模型收斂穩定性)
    if actor_losses and critic_losses:
        fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(actor_losses, color='green')
        ax1.set_title("Actor Loss")
        ax1.grid(True)
        
        ax2.plot(critic_losses, color='red')
        ax2.set_title("Critic Loss")
        ax2.grid(True)
        plt.tight_layout()
        save_plot(fig4, "v4_losses.png")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()