import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 會依序嘗試這些路徑，方便你直接用目前上傳檔案或專案裡的 results 檔案
CANDIDATE_FILES = {
    "V9": [
        "td3_v9_training_metrics(1).json",
        "td3_v9_training_metrics.json",
        "results/td3_v9_training_metrics.json",
    ],
    "V10": [
        "td3_v10_training_metrics(1).json",
        "td3_v10_training_metrics.json",
        "results/td3_v10_training_metrics.json",
    ],
    "V11": [
        "td3_v11_training_metrics(1).json",
        "td3_v11_training_metrics.json",
        "results/td3_v11_training_metrics.json",
    ],
}

OUTPUT_DIR = "results/figures_compare_v9_v10_v11"
SMOOTH_WINDOW = 15

COLORS_VER = {"V9": "lightcoral", "V10": "steelblue", "V11": "seagreen"}
STYLES_VER = {"V9": "--", "V10": "-.", "V11": "-"}
WIDTHS_VER = {"V9": 1.8, "V10": 1.8, "V11": 2.2}

SINGLE_CFG = [
    ("episode_rewards",                "Cumulative Reward",          "compare_reward.png"),
    ("episode_avg_delay",              "Avg Delay (ms)",             "compare_delay.png"),
    ("episode_timeout_ratio",          "Timeout Ratio",              "compare_timeout.png"),
    ("episode_avg_cpu_viol",           "Avg CPU Violation",          "compare_cpu_viol.png"),
    ("episode_avg_deadline_pressure",  "Deadline Pressure",          "compare_deadline_pressure.png"),
    ("episode_avg_sinr",               "Avg SINR",                   "compare_sinr.png"),
    ("episode_avg_channel_rate",       "Avg Channel Rate R (Mbps)",  "compare_channel_rate.png"),
    ("episode_channel_overflow_ratio", "Channel Overflow Ratio",     "compare_ch_overflow.png"),
    ("episode_avg_channel_entropy",    "Channel Assignment Entropy", "compare_ch_entropy.png"),
    ("episode_avg_rho",                "Avg Offload Ratio ρ",        "compare_rho.png"),
    ("episode_avg_queue_delta",        "Avg Queue Delta |Δq|",       "compare_queue_delta.png"),
    ("episode_avg_t_ul",               "Avg Upload Delay t_ul",      "compare_t_ul.png"),
    ("episode_avg_t_comp",             "Avg Compute Delay t_comp",   "compare_t_comp.png"),
    ("episode_avg_t_link",             "Avg Link Delay t_link",      "compare_t_link.png"),
]



def resolve_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None



def load_versions():
    loaded = {}
    resolved_paths = {}
    for ver, candidates in CANDIDATE_FILES.items():
        fpath = resolve_file(candidates)
        if fpath is None:
            print(f"⚠️ 找不到 {ver} metrics 檔案，略過")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            loaded[ver] = json.load(f)
        resolved_paths[ver] = fpath
        print(f"✅ 載入 {ver}: {fpath}")
    return loaded, resolved_paths



def smooth(data, window=SMOOTH_WINDOW):
    out = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        out.append(sum(data[lo:i + 1]) / (i - lo + 1))
    return out



def plot_compare_line(ax, loaded, key, ylabel, show_raw=False):
    plotted = False
    for ver in ["V9", "V10", "V11"]:
        if ver not in loaded or key not in loaded[ver]:
            continue
        vals = loaded[ver][key]
        eps = list(range(len(vals)))

        if show_raw:
            ax.plot(eps, vals, color=COLORS_VER[ver], alpha=0.12, linewidth=0.7)

        ax.plot(
            eps,
            smooth(vals),
            color=COLORS_VER[ver],
            linestyle=STYLES_VER[ver],
            linewidth=WIDTHS_VER[ver],
            label=ver,
        )
        plotted = True

    if plotted:
        ax.set_title(ylabel, fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(fontsize=10)

    return plotted



def save_single_compare_plots(loaded, output_dir):
    print("\n📈 輸出每個指標的 V9/V10/V11 比較圖...")
    for key, ylabel, fname in SINGLE_CFG:
        if not any(ver in loaded and key in loaded[ver] for ver in loaded):
            print(f"⚠️ {key} 不存在於任何版本，略過")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = plot_compare_line(ax, loaded, key, ylabel, show_raw=True)
        if not plotted:
            plt.close(fig)
            continue

        ax.set_title(f"V9 vs V10 vs V11 — {ylabel}", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close(fig)
        print(f"✅ {fname}")



def save_delay_decomp_compare(loaded, output_dir):
    decomp_keys = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
    decomp_labels = ["Upload t_ul (NOMA × ρ)", "Compute t_comp", "Link t_link"]
    decomp_colors = ["#4C9BE8", "#E87B4C", "#6DBF67"]

    available_versions = [
        ver for ver in ["V9", "V10", "V11"]
        if ver in loaded and all(k in loaded[ver] for k in decomp_keys)
    ]

    if not available_versions:
        print("⚠️ 沒有版本同時具備 delay decomposition 三個欄位，略過")
        return

    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, len(available_versions), figure=fig, wspace=0.25)

    for idx, ver in enumerate(available_versions):
        ax = fig.add_subplot(gs[0, idx])
        eps = list(range(len(loaded[ver][decomp_keys[0]])))
        bottoms = np.zeros(len(eps))

        for key, label, color in zip(decomp_keys, decomp_labels, decomp_colors):
            vals = np.array(smooth(loaded[ver][key]))
            ax.fill_between(eps, bottoms, bottoms + vals, alpha=0.75, color=color, label=label)
            bottoms += vals

        ax.set_title(ver, fontsize=12)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Delay (ms)")
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx == len(available_versions) - 1:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("V9 vs V10 vs V11 — Delay Decomposition (smoothed)", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(output_dir, "compare_delay_decomp.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ compare_delay_decomp.png")



def save_noma_panel_compare(loaded, output_dir):
    noma_cfg = [
        ("episode_avg_sinr",               "Avg SINR"),
        ("episode_avg_channel_rate",       "Avg Channel Rate (Mbps)"),
        ("episode_channel_overflow_ratio", "Channel Overflow Ratio"),
        ("episode_avg_channel_entropy",    "Channel Entropy"),
    ]

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    for idx, (key, ylabel) in enumerate(noma_cfg):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        plotted = plot_compare_line(ax, loaded, key, ylabel, show_raw=False)
        if not plotted:
            ax.set_visible(False)

    fig.suptitle("TD3 V9 vs V10 vs V11 — NOMA Channel Metrics", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(os.path.join(output_dir, "compare_noma_panel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ compare_noma_panel.png")



def save_main_panel_compare(loaded, output_dir):
    panel_cfg = [
        ("episode_rewards",               "Cumulative Reward"),
        ("episode_avg_delay",             "Avg Delay (ms)"),
        ("episode_timeout_ratio",         "Timeout Ratio"),
        ("episode_avg_cpu_viol",          "Avg CPU Violation"),
        ("episode_avg_rho",               "Avg Offload Ratio ρ"),
        ("episode_avg_channel_rate",      "Avg Channel Rate"),
    ]

    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    for idx, (key, ylabel) in enumerate(panel_cfg):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        plotted = plot_compare_line(ax, loaded, key, ylabel, show_raw=False)
        if not plotted:
            ax.set_visible(False)

    fig.suptitle("TD3 V9 vs V10 vs V11 — Main Training Metrics", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(output_dir, "compare_main_panel.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ compare_main_panel.png")



def print_summary(loaded):
    report = [
        ("episode_avg_delay",              "Avg Delay"),
        ("episode_timeout_ratio",          "Timeout Ratio"),
        ("episode_avg_cpu_viol",           "Avg CPU Viol"),
        ("episode_avg_deadline_pressure",  "Deadline Pressure"),
        ("episode_avg_t_ul",               "Avg t_ul"),
        ("episode_avg_t_comp",             "Avg t_comp"),
        ("episode_avg_t_link",             "Avg t_link"),
        ("episode_avg_sinr",               "Avg SINR"),
        ("episode_avg_channel_rate",       "Avg Channel Rate"),
        ("episode_channel_overflow_ratio", "Channel Overflow Ratio"),
        ("episode_avg_channel_entropy",    "Channel Entropy"),
        ("episode_avg_rho",                "Avg Rho"),
        ("episode_avg_queue_delta",        "Avg Queue Delta"),
        ("episode_rewards",                "Avg Reward"),
    ]

    print("\n📊 各版本最後 20 episodes 平均：")
    for ver in ["V9", "V10", "V11"]:
        if ver not in loaded:
            continue
        print(f"\n[{ver}]")
        data = loaded[ver]
        for key, label in report:
            if key in data and len(data[key]) >= 20:
                tail = data[key][-20:]
                print(f"  {label:<22}: {np.mean(tail):>10.4f} ± {np.std(tail):.4f}")



def main():
    loaded, resolved_paths = load_versions()

    if "V11" not in loaded:
        print("❌ 至少需要 V11 metrics 檔案")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_single_compare_plots(loaded, OUTPUT_DIR)
    save_delay_decomp_compare(loaded, OUTPUT_DIR)
    save_noma_panel_compare(loaded, OUTPUT_DIR)
    save_main_panel_compare(loaded, OUTPUT_DIR)
    print_summary(loaded)

    print("\n📁 輸出資料夾:", OUTPUT_DIR)
    print("📄 使用檔案:")
    for ver in ["V9", "V10", "V11"]:
        if ver in resolved_paths:
            print(f"  {ver}: {resolved_paths[ver]}")


if __name__ == "__main__":
    main()
