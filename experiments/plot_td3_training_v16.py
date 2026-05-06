import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUT_FILE = "results/td3_v16_training_metrics.json"
OUTPUT_DIR = "results/figures_v16"
V15_FILE   = "results/td3_v15_training_metrics.json"
V15_FILE   = "results/td3_v15_training_metrics.json"


def smooth(data, window=15):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Cannot find {INPUT_FILE} -- run train_td3_v16.py first")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps = list(range(len(data["episode_rewards"])))

    # ── 1. Individual metric plots ────────────────────────────────────────
    single_cfg = [
        ("episode_rewards",               "Cumulative Reward",             "v16_reward.png",            "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",                "v16_delay.png",             "orangered"),
        ("episode_avg_slack",             "Avg Slack (ms)",                "v16_slack.png",             "#2CA02C"),
        ("episode_timeout_ratio",         "Timeout Ratio",                 "v16_timeout.png",           "black"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",             "v16_cpu_viol.png",          "purple"),
        ("episode_avg_deadline_pressure", "Deadline Pressure",             "v16_deadline_pressure.png", "darkgreen"),
        ("episode_task_mix_urgent_ratio", "Urgent Task Ratio",             "v16_urgent_ratio.png",      "#D62728"),
        ("episode_avg_sinr_db",           "Avg SINR (dB)",                 "v16_sinr.png",              "#1A6FA8"),
        ("episode_avg_channel_rate",      "Avg Channel Rate R (Mbps)",     "v16_channel_rate.png",      "#0F6E56"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio",        "v16_ch_overflow.png",       "#D85A30"),
        ("episode_avg_channel_entropy",   "Channel Assignment Entropy",    "v16_ch_entropy.png",        "#7F77DD"),
        ("episode_avg_rho",               "Avg Offload Ratio rho",         "v16_rho.png",               "#B8860B"),
        ("episode_avg_queue_delta",       "Avg Queue Delta |dq|",          "v16_queue_delta.png",       "#8B4513"),
    ]

    for key, ylabel, fname, color in single_cfg:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, data[key], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0, label="smoothed (w=15)")
        ax.set_title(f"V16 Training -- {ylabel}", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close(fig)
        print(f"saved {fname}")

    # ── 2. Delay decomposition stacked area ──────────────────────────────
    decomp_keys   = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
    decomp_labels = ["Upload t_ul (NOMA x rho)", "Compute t_comp", "Link t_link"]
    decomp_colors = ["#4C9BE8", "#E87B4C", "#6DBF67"]

    if all(k in data for k in decomp_keys):
        fig, ax = plt.subplots(figsize=(12, 5))
        bottoms = np.zeros(len(eps))
        for key, label, color in zip(decomp_keys, decomp_labels, decomp_colors):
            vals = np.array(smooth(data[key]))
            ax.fill_between(eps, bottoms, bottoms + vals, alpha=0.75, color=color, label=label)
            bottoms += vals
        ax.set_title("V16 Training -- Delay Decomposition (smoothed)", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Delay (ms)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "v16_delay_decomp.png"), dpi=300)
        plt.close(fig)
        print("saved v16_delay_decomp.png")

    # ── 3. NOMA 4-panel ──────────────────────────────────────────────────
    noma_cfg = [
        ("episode_avg_sinr_db",           "Avg SINR (dB)",           "#1A6FA8"),
        ("episode_avg_channel_rate",      "Avg Channel Rate (Mbps)", "#0F6E56"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio",  "#D85A30"),
        ("episode_avg_channel_entropy",   "Channel Entropy",         "#7F77DD"),
    ]
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)
    for idx, (key, ylabel, color) in enumerate(noma_cfg):
        if key not in data:
            continue
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.plot(eps, data[key], color="lightgray", alpha=0.3, linewidth=0.7)
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("TD3 V16 -- NOMA Channel Metrics (t_link penalty + overflow penalty=15)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(OUTPUT_DIR, "v16_noma_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved v16_noma_panel.png")

    # ── 4. Main 6-panel overview ─────────────────────────────────────────
    panel_cfg = [
        ("episode_rewards",               "Cumulative Reward",      "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",         "orangered"),
        ("episode_timeout_ratio",         "Timeout Ratio",          "dimgray"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",      "purple"),
        ("episode_avg_rho",               "Avg Offload Ratio rho",  "#B8860B"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio", "#D85A30"),
    ]
    fig = plt.figure(figsize=(16, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)
    for idx, (key, ylabel, color) in enumerate(panel_cfg):
        if key not in data:
            continue
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.plot(eps, data[key], color="lightgray", alpha=0.3, linewidth=0.7)
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("TD3 V16 Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v16_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved v16_panel.png")

    # ── 5. V14 / V15 / V16 comparison ────────────────────────────────────
    compare_cfg = [
        ("episode_rewards",               "Cumulative Reward"),
        ("episode_avg_delay",             "Avg Delay (ms)"),
        ("episode_avg_t_link",            "Avg t_link (ms)"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio"),
        ("episode_avg_rho",               "Avg Offload Ratio rho"),
        ("episode_avg_t_ul",              "Avg t_ul (ms)"),
    ]
    loaded = {}
    for label, fpath in [("V14", V14_FILE), ("V15", V15_FILE)]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                loaded[label] = json.load(f)

    colors_ver = {"V14": "lightcoral", "V15": "steelblue", "V16": "seagreen"}
    styles_ver = {"V14": "--",          "V15": "-.",         "V16": "-"}

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)
    for idx, (key, ylabel) in enumerate(compare_cfg):
        if key not in data:
            continue
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        for ver, vdata in loaded.items():
            if key not in vdata:
                continue
            ve = list(range(len(vdata[key])))
            ax.plot(ve, smooth(vdata[key]), color=colors_ver[ver],
                    linewidth=1.5, linestyle=styles_ver[ver], label=ver, alpha=0.8)
        ax.plot(eps, smooth(data[key]), color=colors_ver["V16"],
                linewidth=2.2, linestyle=styles_ver["V16"], label="V16")
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("V14 vs V15 vs V16 -- Training Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v14_v15_v16_compare.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved v14_v15_v16_compare.png")

    # ── 6. Numeric summary (last 20 episodes) ────────────────────────────
    print("\nV16 Training Summary (last 20 episodes mean +/- std):")
    report = [
        ("episode_avg_delay",             "Avg Delay             "),
        ("episode_avg_slack",             "Avg Slack             "),
        ("episode_timeout_ratio",         "Timeout Ratio         "),
        ("episode_avg_cpu_viol",          "Avg CPU Viol          "),
        ("episode_avg_deadline_pressure", "Deadline Pressure     "),
        ("episode_task_mix_urgent_ratio", "Urgent Task Ratio     "),
        ("episode_avg_t_ul",              "Avg t_ul              "),
        ("episode_avg_t_comp",            "Avg t_comp            "),
        ("episode_avg_t_link",            "Avg t_link            "),
        ("episode_avg_sinr_db",           "Avg SINR (dB)         "),
        ("episode_avg_channel_rate",      "Avg Channel Rate      "),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio"),
        ("episode_avg_channel_entropy",   "Channel Entropy       "),
        ("episode_avg_rho",               "Avg Rho               "),
        ("episode_avg_queue_delta",       "Avg Queue Delta       "),
        ("episode_rewards",               "Avg Reward            "),
    ]
    for key, label in report:
        if key in data and len(data[key]) >= 20:
            tail = data[key][-20:]
            print(f"  {label}: {np.mean(tail):.4f}  +/- {np.std(tail):.4f}")


if __name__ == "__main__":
    main()
