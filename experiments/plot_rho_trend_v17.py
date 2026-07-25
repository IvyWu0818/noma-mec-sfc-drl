"""
plot_rho_trend_v17.py
Plot average offload ratio (rho) vs. training progress from a TD3(V17)
training run's metrics JSON (episode_avg_rho, logged per-episode by
V17MetricsCallback in agents/train_td3_v17.py). Binned over every N episodes
to smooth the per-episode noise, so we can see whether the agent learns to
raise rho over training (favoring the offload branch, t_offload) or keeps
rho low (relying on the local-compute branch, t_local).

Run: python -m experiments.plot_rho_trend_v17
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

INPUT_JSON = "results/td3_v17_training_metrics.json"
OUTPUT_DIR = "results/figures_v17"
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "rho_trend_v17.png")
BIN_SIZE   = 50   # episodes per bin

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(INPUT_JSON) as f:
        metrics = json.load(f)

    rho = np.array(metrics["episode_avg_rho"])
    delay = np.array(metrics["episode_avg_delay"])
    n_episodes = len(rho)
    n_bins = n_episodes // BIN_SIZE

    rho_binned = rho[:n_bins * BIN_SIZE].reshape(n_bins, BIN_SIZE).mean(axis=1)
    rho_std_binned = rho[:n_bins * BIN_SIZE].reshape(n_bins, BIN_SIZE).std(axis=1)
    delay_binned = delay[:n_bins * BIN_SIZE].reshape(n_bins, BIN_SIZE).mean(axis=1)
    bin_centers = (np.arange(n_bins) + 0.5) * BIN_SIZE

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(bin_centers, rho_binned, color="#2E86AB", lw=1.8)
    ax1.fill_between(bin_centers, rho_binned - rho_std_binned, rho_binned + rho_std_binned,
                      color="#2E86AB", alpha=0.15)
    ax1.axhline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.6, label="rho=1 (全卸載)")
    ax1.set_ylabel(f"平均 rho（每 {BIN_SIZE} episodes 平均）", fontsize=10)
    ax1.set_title("TD3(V17) 訓練過程：平均卸載比例 rho vs. Episode", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=9)

    ax2.plot(bin_centers, delay_binned, color="#E07B54", lw=1.8)
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel(f"平均延遲 ms（每 {BIN_SIZE} episodes 平均）", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300)
    fig.savefig(os.path.splitext(OUTPUT_PNG)[0] + ".svg")
    print(f"saved {OUTPUT_PNG} (+ .svg)")

    print(f"\n  rho: first bin avg={rho_binned[0]:.4f}   last bin avg={rho_binned[-1]:.4f}")
    print(f"  delay: first bin avg={delay_binned[0]:.4f}ms   last bin avg={delay_binned[-1]:.4f}ms")


if __name__ == "__main__":
    main()
