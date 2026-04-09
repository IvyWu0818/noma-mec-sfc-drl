import os
import json
import matplotlib.pyplot as plt

INPUT_FILE = "results/td3_v5_training_metrics.json"
OUTPUT_DIR = "results/figures_v5"

def smooth(data, window=15):
    return [sum(data[max(0, i-window+1):i+1]) / (i - max(0, i-window+1) + 1) for i in range(len(data))]

def main():
    if not os.path.exists(INPUT_FILE): return
    with open(INPUT_FILE, "r") as f: data = json.load(f)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = [("episode_rewards", "Reward", "v5_reward.png", "blue"), 
               ("episode_avg_delay", "Delay (ms)", "v5_delay.png", "orangered"),
               ("episode_timeout_ratio", "Timeout Ratio", "v5_timeout.png", "black")]

    for key, ylabel, fname, color in metrics:
        fig = plt.figure(figsize=(10, 6))
        plt.plot(data[key], color='lightgray', alpha=0.4)
        plt.plot(smooth(data[key]), color=color, linewidth=2)
        plt.title(f"V5 Stress Test: {ylabel}")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.6)
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        print(f"✅ 已存檔: {fname}")

if __name__ == "__main__":
    main()