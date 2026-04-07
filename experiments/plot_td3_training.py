import os
import json
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

OUTPUT_DIR = "results/figures"
SHOW_PLOTS = False   # 👉 要不要顯示視窗（報告用通常 False）


# ============================================================
# 工具：平滑
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
# 存圖
# ============================================================

def save_plot(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")


# ============================================================
# 主程式
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open("results/td3_v3_training_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    rewards = data["episode_rewards"]
    delays = data["episode_avg_delay"]
    slacks = data["episode_avg_slack"]
    timeouts = data["episode_timeout_ratio"]
    actor_losses = data["actor_losses"]
    critic_losses = data["critic_losses"]

    # ========================================================
    # 1. Reward
    # ========================================================
    fig = plt.figure()
    plt.plot(rewards, label="raw")
    plt.plot(smooth(rewards), label="smoothed")
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    save_plot(fig, "reward.png")

    # ========================================================
    # 2. Delay
    # ========================================================
    fig = plt.figure()
    plt.plot(delays, label="raw")
    plt.plot(smooth(delays), label="smoothed")
    plt.title("Avg Delay per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Delay")
    plt.legend()
    plt.grid()

    save_plot(fig, "delay.png")

    # ========================================================
    # 3. Slack
    # ========================================================
    fig = plt.figure()
    plt.plot(slacks, label="raw")
    plt.plot(smooth(slacks), label="smoothed")
    plt.title("Avg Slack per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Slack")
    plt.legend()
    plt.grid()

    save_plot(fig, "slack.png")

    # ========================================================
    # 4. Timeout Ratio
    # ========================================================
    fig = plt.figure()
    plt.plot(timeouts, label="raw")
    plt.plot(smooth(timeouts), label="smoothed")
    plt.title("Timeout Ratio per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid()

    save_plot(fig, "timeout.png")

    # ========================================================
    # 5. Critic Loss
    # ========================================================
    if critic_losses:
        fig = plt.figure()
        plt.plot(critic_losses)
        plt.title("Critic Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid()

        save_plot(fig, "critic_loss.png")

    # ========================================================
    # 6. Actor Loss
    # ========================================================
    if actor_losses:
        fig = plt.figure()
        plt.plot(actor_losses)
        plt.title("Actor Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid()

        save_plot(fig, "actor_loss.png")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()