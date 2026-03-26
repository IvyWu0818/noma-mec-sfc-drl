import json
import matplotlib.pyplot as plt


def smooth(data, window=10):
    if not data:
        return []
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(sum(data[start:i + 1]) / (i - start + 1))
    return smoothed


def main():
    with open("results/td3_v3_training_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    episode_rewards = data["episode_rewards"]
    episode_avg_delay = data["episode_avg_delay"]
    episode_avg_slack = data["episode_avg_slack"]
    episode_timeout_ratio = data["episode_timeout_ratio"]
    actor_losses = data["actor_losses"]
    critic_losses = data["critic_losses"]

    # 1. Episode reward
    plt.figure()
    plt.plot(episode_rewards, label="raw")
    plt.plot(smooth(episode_rewards, window=10), label="smoothed")
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    # 2. Avg delay
    plt.figure()
    plt.plot(episode_avg_delay, label="raw")
    plt.plot(smooth(episode_avg_delay, window=10), label="smoothed")
    plt.title("Episode Avg Delay")
    plt.xlabel("Episode")
    plt.ylabel("Delay")
    plt.legend()
    plt.grid()

    # 3. Avg slack
    plt.figure()
    plt.plot(episode_avg_slack, label="raw")
    plt.plot(smooth(episode_avg_slack, window=10), label="smoothed")
    plt.title("Episode Avg Slack")
    plt.xlabel("Episode")
    plt.ylabel("Slack")
    plt.legend()
    plt.grid()

    # 4. Timeout ratio
    plt.figure()
    plt.plot(episode_timeout_ratio, label="raw")
    plt.plot(smooth(episode_timeout_ratio, window=10), label="smoothed")
    plt.title("Episode Timeout Ratio")
    plt.xlabel("Episode")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid()

    # 5. Critic loss
    if critic_losses:
        plt.figure()
        plt.plot(critic_losses)
        plt.title("Critic Loss")
        plt.xlabel("Training Update")
        plt.ylabel("Loss")
        plt.grid()

    # 6. Actor loss
    if actor_losses:
        plt.figure()
        plt.plot(actor_losses)
        plt.title("Actor Loss")
        plt.xlabel("Training Update")
        plt.ylabel("Loss")
        plt.grid()

    plt.show()


if __name__ == "__main__":
    main()