import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

INPUT_FILE = "results/td3_v13_training_metrics.json"
OUTPUT_DIR = "results/figures_v13"
V11_FILE   = "results/td3_v11_training_metrics.json"
V12_FILE   = "results/td3_v12_training_metrics.json"


def smooth(data, window=15):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Cannot find {INPUT_FILE} -- run train_td3_v13.py first")
        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps = list(range(len(data["episode_rewards"])))

    # ── 1. Individual metric plots ────────────────────────────────────────
    single_cfg = [
        ("episode_rewards",               "Cumulative Reward",             "v13_reward.png",            "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",                "v13_delay.png",             "orangered"),
        ("episode_avg_slack",             "Avg Slack (ms)",                "v13_slack.png",             "#2CA02C"),
        ("episode_timeout_ratio",         "Timeout Ratio",                 "v13_timeout.png",           "black"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",             "v13_cpu_viol.png",          "purple"),
        ("episode_avg_deadline_pressure", "Deadline Pressure",             "v13_deadline_pressure.png", "darkgreen"),
        ("episode_task_mix_urgent_ratio", "Urgent Task Ratio",             "v13_urgent_ratio.png",      "#D62728"),
        ("episode_avg_sinr_db",           "Avg SINR (dB)",                 "v13_sinr.png",              "#1A6FA8"),
        ("episode_avg_channel_rate",      "Avg Channel Rate R (Mbps)",     "v13_channel_rate.png",      "#0F6E56"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio",        "v13_ch_overflow.png",       "#D85A30"),
        ("episode_avg_channel_entropy",   "Channel Assignment Entropy",    "v13_ch_entropy.png",        "#7F77DD"),
        ("episode_avg_rho",               "Avg Offload Ratio rho",         "v13_rho.png",               "#B8860B"),
        ("episode_avg_queue_delta",       "Avg Queue Delta |dq|",          "v13_queue_delta.png",       "#8B4513"),
    ]

    for key, ylabel, fname, color in single_cfg:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, data[key], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
        ax.plot(eps, smooth(data[key]), color=color, linewidth=2.0, label="smoothed (w=15)")
        ax.set_title(f"V13 Training -- {ylabel}", fontsize=13)
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
        ax.set_title("V13 Training -- Delay Decomposition (smoothed)", fontsize=13)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Delay (ms)")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "v13_delay_decomp.png"), dpi=300)
        plt.close(fig)
        print("saved v13_delay_decomp.png")

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
    fig.suptitle("TD3 V13 -- NOMA Channel Metrics (5G NR params + V11 obs)", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.savefig(os.path.join(OUTPUT_DIR, "v13_noma_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved v13_noma_panel.png")

    # ── 4. Main 6-panel overview ─────────────────────────────────────────
    panel_cfg = [
        ("episode_rewards",               "Cumulative Reward",      "steelblue"),
        ("episode_avg_delay",             "Avg Delay (ms)",         "orangered"),
        ("episode_timeout_ratio",         "Timeout Ratio",          "dimgray"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation",      "purple"),
        ("episode_avg_rho",               "Avg Offload Ratio rho",  "#B8860B"),
        ("episode_avg_channel_rate",      "Avg Channel Rate (Mbps)","#0F6E56"),
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
    fig.suptitle("TD3 V13 Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v13_panel.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved v13_panel.png")

    # ── 5. V11 / V12 / V13 comparison ────────────────────────────────────
    compare_cfg = [
        ("episode_rewards",               "Cumulative Reward"),
        ("episode_avg_delay",             "Avg Delay (ms)"),
        ("episode_channel_overflow_ratio","Channel Overflow Ratio"),
        ("episode_timeout_ratio",         "Timeout Ratio"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation"),
        ("episode_avg_rho",               "Avg Offload Ratio rho"),
    ]
    loaded = {}
    for label, fpath in [("V11", V11_FILE), ("V12", V12_FILE)]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                loaded[label] = json.load(f)

    colors_ver = {"V11": "lightcoral", "V12": "steelblue", "V13": "seagreen"}
    styles_ver = {"V11": "--",          "V12": "-.",         "V13": "-"}

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
        ax.plot(eps, smooth(data[key]), color=colors_ver["V13"],
                linewidth=2.2, linestyle=styles_ver["V13"], label="V13")
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("V11 vs V12 vs V13 -- Training Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUTPUT_DIR, "v11_v12_v13_compare.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved v11_v12_v13_compare.png")

    # ── 6. Numeric summary (last 20 episodes) ────────────────────────────
    print("\nV13 Training Summary (last 20 episodes mean +/- std):")
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
