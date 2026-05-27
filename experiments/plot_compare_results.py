"""
plot_compare_results.py
V14 vs V17 Training Comparison -- 6-panel figure
Panels:
  1. Cumulative Reward
  2. End-to-End Delay (ms)
  3. Delay Decomposition (stacked area: t_ul / t_comp / t_link)
  4. Avg Channel Rate (Mbps)
  5. Deadline Pressure
  6. CPU Violation Rate
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Settings ──────────────────────────────────────────────────────────────
V14_FILE   = "results/td3_v14_training_metrics.json"
V17_FILE   = "results/td3_v17_training_metrics.json"
OUTPUT_DIR = "results/figures_compare"

# V14 cpu_viol is raw MCycles/ms; normalize to rate for fair comparison
TOTAL_MEC_CPU_CAP = 135.0   # mec0=35 + mec1=45 + mec2=55 MCycles/ms

# Visual style
COLORS     = {"V14": "#E07B54", "V17": "#4A90D9"}
STYLES     = {"V14": "--",       "V17": "-"}
LW         = {"V14": 1.8,        "V17": 2.2}
ALPHA_RAW  = 0.18
SMOOTH_W   = 20


def smooth(data, window=SMOOTH_W):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out


def load(fpath):
    with open(fpath) as f:
        return json.load(f)


def add_line(ax, eps, data, ver, key, show_raw=True):
    if key not in data:
        return None
    if show_raw:
        ax.plot(eps, data[key], color=COLORS[ver], alpha=ALPHA_RAW, linewidth=0.6)
    h, = ax.plot(eps, smooth(data[key]),
                 color=COLORS[ver], linewidth=LW[ver],
                 linestyle=STYLES[ver], label=ver)
    return h


def main():
    if not os.path.exists(V14_FILE):
        print(f"Missing {V14_FILE}"); return
    if not os.path.exists(V17_FILE):
        print(f"Missing {V17_FILE}"); return

    d14 = load(V14_FILE)
    d17 = load(V17_FILE)

    # V14: normalize cpu_viol raw -> rate
    if "episode_avg_cpu_viol" in d14:
        d14["episode_cpu_viol_rate"] = [
            v / TOTAL_MEC_CPU_CAP for v in d14["episode_avg_cpu_viol"]
        ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eps14 = list(range(len(d14["episode_rewards"])))
    eps17 = list(range(len(d17["episode_rewards"])))

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.34)

    # ── Panel 1: Cumulative Reward ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for ver, eps, d in [("V14", eps14, d14), ("V17", eps17, d17)]:
        add_line(ax1, eps, d, ver, "episode_rewards")
    ax1.set_title("Cumulative Reward", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=9)
    ax1.set_ylabel("Reward", fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 2: End-to-End Delay ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for ver, eps, d in [("V14", eps14, d14), ("V17", eps17, d17)]:
        add_line(ax2, eps, d, ver, "episode_avg_delay")
    ax2.set_title("End-to-End Delay (ms)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode", fontsize=9)
    ax2.set_ylabel("Avg Delay (ms)", fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 3: Delay Decomposition ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    decomp_keys   = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
    decomp_labels = ["t_ul (upload)", "t_comp (compute)", "t_link (backhaul)"]
    decomp_colors_v17 = ["#7DC3E8", "#F0A070", "#82C882"]
    decomp_colors_v14 = ["#B8D8F0", "#F8C8A8", "#B8E4B8"]

    for (d, eps, dc, alpha, lbl_suffix) in [
        (d14, eps14, decomp_colors_v14, 0.45, " (V14)"),
        (d17, eps17, decomp_colors_v17, 0.72, " (V17)"),
    ]:
        if all(k in d for k in decomp_keys):
            bottoms = np.zeros(len(eps))
            for key, label, color in zip(decomp_keys, decomp_labels, dc):
                vals = np.array(smooth(d[key]))
                ax3.fill_between(eps, bottoms, bottoms + vals,
                                 alpha=alpha, color=color,
                                 label=label + lbl_suffix)
                bottoms += vals

    ax3.set_title("Delay Decomposition", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Episode", fontsize=9)
    ax3.set_ylabel("Delay (ms)", fontsize=9)
    ax3.legend(fontsize=7, loc="upper right", ncol=2)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 4: Avg Channel Rate ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for ver, eps, d in [("V14", eps14, d14), ("V17", eps17, d17)]:
        add_line(ax4, eps, d, ver, "episode_avg_channel_rate")
    ax4.set_title("Avg Channel Rate (Mbps)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Episode", fontsize=9)
    ax4.set_ylabel("Rate (Mbps)", fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 5: Deadline Pressure ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for ver, eps, d in [("V14", eps14, d14), ("V17", eps17, d17)]:
        add_line(ax5, eps, d, ver, "episode_avg_deadline_pressure")
    ax5.set_title("Deadline Pressure", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Episode", fontsize=9)
    ax5.set_ylabel("Pressure (delay / deadline)", fontsize=9)
    ax5.legend(fontsize=10)
    ax5.grid(True, linestyle="--", alpha=0.4)

    # ── Panel 6: CPU Violation Rate ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    for ver, eps, d in [("V14", eps14, d14), ("V17", eps17, d17)]:
        add_line(ax6, eps, d, ver, "episode_cpu_viol_rate")
    ax6.set_title("CPU Violation Rate", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Episode", fontsize=9)
    ax6.set_ylabel("CPU Violation Rate (raw / 135)", fontsize=9)
    ax6.legend(fontsize=10)
    ax6.grid(True, linestyle="--", alpha=0.4)

    # ── Main title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "TD3 Training: V14 vs V17  |  smoothing window = 20 eps\n"
        "V14: reward_scale=50  |  V17: reward_scale=75  "
        "(reward values not directly comparable)",
        fontsize=12, fontweight="bold", y=1.02
    )

    out_path = os.path.join(OUTPUT_DIR, "v14_v17_compare_6panel.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # ── Numeric summary table ──────────────────────────────────────────────
    print("\n--- Last 20 episodes summary (mean +/- std) ---")
    metrics = [
        ("episode_rewards",               "Reward             "),
        ("episode_avg_delay",             "Delay (ms)         "),
        ("episode_avg_t_ul",              "  t_ul (ms)        "),
        ("episode_avg_t_comp",            "  t_comp (ms)      "),
        ("episode_avg_t_link",            "  t_link (ms)      "),
        ("episode_avg_channel_rate",      "CH Rate (Mbps)     "),
        ("episode_avg_deadline_pressure", "Deadline Pressure  "),
        ("episode_cpu_viol_rate",         "CPU Viol Rate      "),
        ("episode_channel_overflow_ratio","CH Overflow Ratio  "),
        ("episode_avg_rho",               "Avg Rho            "),
        ("episode_timeout_ratio",         "Timeout Ratio      "),
    ]
    print(f"  {'Metric':<22}  {'V14':>22}  {'V17':>22}")
    print("  " + "-" * 70)
    for key, label in metrics:
        v14_tail = d14.get(key, [])[-20:]
        v17_tail = d17.get(key, [])[-20:]
        s14 = f"{np.mean(v14_tail):.4f} +/- {np.std(v14_tail):.4f}" if len(v14_tail) >= 20 else "N/A"
        s17 = f"{np.mean(v17_tail):.4f} +/- {np.std(v17_tail):.4f}" if len(v17_tail) >= 20 else "N/A"
        print(f"  {label}  {s14:>22}  {s17:>22}")


if __name__ == "__main__":
    main()
