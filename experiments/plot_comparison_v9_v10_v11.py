"""
plot_comparison_v9_v10_v11.py
─────────────────────────────
使用三個版本訓練摘要（最後 20 episodes 平均值）繪製完整比較圖。
數值來源：各版本訓練結束後的 console 摘要輸出。

執行：
    python plot_comparison_v9_v10_v11.py
輸出：
    results/comparison/  ─ 所有比較圖
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = "results/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 數據（最後 20 episodes 平均 ± std）
# ══════════════════════════════════════════════════════════════════════
DATA = {
    "V9": {
        "Avg Delay (ms)":             (16.70, 0.80),
        "Timeout Ratio":              (0.430, 0.030),
        "Avg CPU Violation":          (0.750, 0.200),
        "Deadline Pressure":          (1.040, 0.040),
        "Avg t_ul (ms)":              (10.40, 0.50),
        "Avg t_comp (ms)":            (3.500, 0.200),
        "Avg t_link (ms)":            (2.900, 0.150),
        "Avg SINR":                   (3.150, 0.850),
        "Avg Channel Rate (Mbps)":    (9.000, 0.860),
        "Channel Overflow Ratio":     (0.956, 0.009),
        "Channel Entropy":            (1.585, 0.000),
        "Avg Rho":                    (1.000, 0.000),   # 固定全卸載
        "Avg Reward":                 (-150.0, 15.0),
    },
    "V10": {
        "Avg Delay (ms)":             (7.802, 0.284),
        "Timeout Ratio":              (0.006, 0.008),
        "Avg CPU Violation":          (0.255, 0.110),
        "Deadline Pressure":          (0.491, 0.017),
        "Avg t_ul (ms)":              (0.656, 0.111),
        "Avg t_comp (ms)":            (4.765, 0.209),
        "Avg t_link (ms)":            (2.382, 0.119),
        "Avg SINR":                   (3.284, 0.854),
        "Avg Channel Rate (Mbps)":    (9.007, 0.865),
        "Channel Overflow Ratio":     (0.956, 0.009),
        "Channel Entropy":            (1.585, 0.000),
        "Avg Rho":                    (0.095, 0.014),
        "Avg Reward":                 (-39.84, 1.185),
    },
    "V11": {
        "Avg Delay (ms)":             (7.408, 0.195),
        "Timeout Ratio":              (0.0035, 0.006),
        "Avg CPU Violation":          (0.422, 0.221),
        "Deadline Pressure":          (0.464, 0.016),
        "Avg t_ul (ms)":              (0.223, 0.036),
        "Avg t_comp (ms)":            (4.211, 0.165),
        "Avg t_link (ms)":            (2.974, 0.102),
        "Avg SINR":                   (29.358, 2.672),
        "Avg Channel Rate (Mbps)":    (36.470, 1.415),
        "Channel Overflow Ratio":     (0.241, 0.042),
        "Channel Entropy":            (1.585, 0.000),
        "Avg Rho":                    (0.165, 0.016),
        "Avg Reward":                 (-27.109, 2.164),
    },
}

VERSIONS = ["V9", "V10", "V11"]
COLORS   = {"V9": "#B4B2A9", "V10": "#378ADD", "V11": "#1D9E75"}
HATCHES  = {"V9": "///", "V10": "", "V11": ""}

def vals(metric):
    return [DATA[v][metric][0] for v in VERSIONS]

def errs(metric):
    return [DATA[v][metric][1] for v in VERSIONS]

def bar_group(ax, metric, title, ylabel=None, pct=False, ylim=None, invert=False):
    """3-group bar chart for a single metric."""
    x  = np.arange(len(VERSIONS))
    v  = np.array(vals(metric))
    e  = np.array(errs(metric))
    cs = [COLORS[vv] for vv in VERSIONS]
    hs = [HATCHES[vv] for vv in VERSIONS]

    bars = ax.bar(x, v, yerr=e, capsize=4, color=cs, hatch=hs,
                  edgecolor="white", linewidth=0.8, error_kw=dict(elinewidth=1, ecolor="#555"))
    ax.set_xticks(x)
    ax.set_xticklabels(VERSIONS, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, vv in zip(bars, v):
        fmt = f"{vv:.1%}" if pct else (f"{vv:.1f}" if vv >= 1 else f"{vv:.3f}")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e[list(vals(metric)).index(vv)] * 0.1 + max(v) * 0.01,
                fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════
# 圖 1：核心效能 4 合 1
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("V9 / V10 / V11 — Core Performance Comparison",
             fontsize=14, fontweight="bold", y=1.01)

bar_group(axes[0,0], "Avg Delay (ms)",    "Avg End-to-End Delay",    "ms",   ylim=(0, 20))
bar_group(axes[0,1], "Timeout Ratio",     "Timeout Ratio",           "ratio",ylim=(0, 0.55))
bar_group(axes[1,0], "Avg CPU Violation", "Avg CPU Violation",       "units",ylim=(0, 1.1))
bar_group(axes[1,1], "Deadline Pressure", "Deadline Pressure",       "delay/ddl", ylim=(0, 1.3))

# 加 deadline = 1 參考線
axes[1,1].axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="deadline limit")
axes[1,1].legend(fontsize=8)

# 版本圖例
legend_patches = [plt.Rectangle((0,0),1,1, color=COLORS[v], label=v) for v in VERSIONS]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/compare_core_performance.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ compare_core_performance.png")


# ══════════════════════════════════════════════════════════════════════
# 圖 2：NOMA 通道指標 4 合 1
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("V9 / V10 / V11 — NOMA Channel Metrics Comparison",
             fontsize=14, fontweight="bold", y=1.01)

bar_group(axes[0,0], "Channel Overflow Ratio",  "Channel Overflow Ratio",   "ratio", ylim=(0, 1.1))
bar_group(axes[0,1], "Avg SINR",                "Avg SINR",                  "",     ylim=(0, 38))
bar_group(axes[1,0], "Avg Channel Rate (Mbps)", "Avg Channel Rate",          "Mbps", ylim=(0, 44))
bar_group(axes[1,1], "Channel Entropy",         "Channel Assignment Entropy","bits", ylim=(1.50, 1.62))

# overflow ratio 加 V9/V10 水平參考線
for ax in [axes[0,0]]:
    ax.axhline(0.956, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="V9/V10 level")
    ax.legend(fontsize=8)

legend_patches = [plt.Rectangle((0,0),1,1, color=COLORS[v], label=v) for v in VERSIONS]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/compare_noma_metrics.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ compare_noma_metrics.png")


# ══════════════════════════════════════════════════════════════════════
# 圖 3：延遲分解堆疊長條圖
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("V9 / V10 / V11 — Delay Decomposition",
             fontsize=14, fontweight="bold")

x     = np.arange(len(VERSIONS))
w     = 0.5
t_ul  = np.array(vals("Avg t_ul (ms)"))
t_comp= np.array(vals("Avg t_comp (ms)"))
t_link= np.array(vals("Avg t_link (ms)"))

b1 = ax.bar(x, t_ul,   w, label="t_ul  (upload)",  color="#4C9BE8", edgecolor="white")
b2 = ax.bar(x, t_comp, w, bottom=t_ul,             label="t_comp (compute)", color="#E87B4C", edgecolor="white")
b3 = ax.bar(x, t_link, w, bottom=t_ul + t_comp,   label="t_link (link)",    color="#6DBF67", edgecolor="white")

# 標上總值
for i, (u, c, l) in enumerate(zip(t_ul, t_comp, t_link)):
    total = u + c + l
    ax.text(i, total + 0.2, f"{total:.2f} ms", ha="center", fontsize=10, fontweight="bold")

# 在每段中間標小數字
for i, (u, c, l) in enumerate(zip(t_ul, t_comp, t_link)):
    if u > 0.3:
        ax.text(i, u/2, f"{u:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(i, u + c/2, f"{c:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(i, u + c + l/2, f"{l:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(VERSIONS, fontsize=12)
ax.set_ylabel("Avg Delay (ms)", fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, 22)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/compare_delay_decomp.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ compare_delay_decomp.png")


# ══════════════════════════════════════════════════════════════════════
# 圖 4：Reward + Rho 雙指標
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("V9 / V10 / V11 — Reward & Offload Ratio",
             fontsize=14, fontweight="bold")

# Reward（負值，越接近 0 越好）
rw = np.array([DATA[v]["Avg Reward"][0] for v in VERSIONS])
re = np.array([DATA[v]["Avg Reward"][1] for v in VERSIONS])
cs = [COLORS[v] for v in VERSIONS]
bars = axes[0].bar(np.arange(3), rw, yerr=re, capsize=4, color=cs,
                   edgecolor="white", error_kw=dict(elinewidth=1, ecolor="#555"))
axes[0].set_xticks([0,1,2])
axes[0].set_xticklabels(VERSIONS, fontsize=11)
axes[0].set_title("Cumulative Reward (higher = better)", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Reward")
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
for bar, v in zip(bars, rw):
    axes[0].text(bar.get_x() + bar.get_width()/2, v - 5,
                 f"{v:.1f}", ha="center", va="top", fontsize=9, fontweight="bold", color="white")

# Rho
bar_group(axes[1], "Avg Rho", "Avg Offload Ratio ρ", "ratio (0~1)", ylim=(0, 1.15))

legend_patches = [plt.Rectangle((0,0),1,1, color=COLORS[v], label=v) for v in VERSIONS]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/compare_reward_rho.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ compare_reward_rho.png")


# ══════════════════════════════════════════════════════════════════════
# 圖 5：雷達圖（正規化後多維比較）
# ══════════════════════════════════════════════════════════════════════
from matplotlib.patches import FancyBboxPatch

RADAR_METRICS = [
    ("Reward↑",            "Avg Reward",              True,  (-150, -27)),
    ("Delay↓",             "Avg Delay (ms)",           False, (0,   17)),
    ("Timeout↓",           "Timeout Ratio",            False, (0,  0.43)),
    ("CPU Viol↓",          "Avg CPU Violation",        False, (0,  0.75)),
    ("SINR↑",              "Avg SINR",                 True,  (3,  30)),
    ("Ch Rate↑",           "Avg Channel Rate (Mbps)",  True,  (9,  37)),
    ("Ch Overflow↓",       "Channel Overflow Ratio",   False, (0,   1.0)),
    ("Deadline P↓",        "Deadline Pressure",        False, (0,   1.1)),
]

labels   = [m[0] for m in RADAR_METRICS]
N        = len(labels)
angles   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles  += angles[:1]

def normalize(key, higher_better, rng, value):
    lo, hi = rng
    if higher_better:
        return (value - lo) / (hi - lo)
    else:
        return 1 - (value - lo) / (hi - lo)

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.suptitle("V9 / V10 / V11 — Radar Comparison\n(normalized, outer = better)",
             fontsize=13, fontweight="bold", y=1.01)

for ver in VERSIONS:
    scores = []
    for lbl, key, hb, rng in RADAR_METRICS:
        v = DATA[ver][key][0]
        scores.append(np.clip(normalize(key, hb, rng, v), 0, 1))
    scores += scores[:1]
    ax.plot(angles, scores, "o-", linewidth=2, label=ver, color=COLORS[ver])
    ax.fill(angles, scores, alpha=0.12, color=COLORS[ver])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.grid(True, linestyle="--", alpha=0.3)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/compare_radar.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ compare_radar.png")


# ══════════════════════════════════════════════════════════════════════
# 圖 6：大型 Summary Dashboard（9 合 1）
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.35)
fig.suptitle("TD3 IIoT — V9 / V10 / V11 Full Training Summary Dashboard",
             fontsize=15, fontweight="bold", y=1.01)

configs = [
    ("Avg Reward",              "Cumulative Reward",        "steelblue",  False, (-165,  -20), True),
    ("Avg Delay (ms)",          "Avg Delay (ms)",           "orangered",  False, (0,     19),  False),
    ("Timeout Ratio",           "Timeout Ratio",            "dimgray",    False, (0,    0.50), False),
    ("Avg CPU Violation",       "Avg CPU Violation",        "purple",     False, (0,    1.0),  False),
    ("Channel Overflow Ratio",  "Channel Overflow Ratio",   "#D85A30",    False, (0,    1.05), False),
    ("Avg SINR",                "Avg SINR",                 "#1A6FA8",    True,  (0,     35),  False),
    ("Avg Channel Rate (Mbps)", "Avg Channel Rate (Mbps)",  "#0F6E56",    True,  (0,     42),  False),
    ("Avg Rho",                 "Avg Offload Ratio ρ",      "#BA7517",    False, (0,    1.1),  False),
    ("Deadline Pressure",       "Deadline Pressure",        "darkgreen",  False, (0,    1.2),  True),
]

for idx, (key, title, color, hb, ylim, ref1) in enumerate(configs):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    x  = np.arange(len(VERSIONS))
    v  = np.array([DATA[vv][key][0] for vv in VERSIONS])
    e  = np.array([DATA[vv][key][1] for vv in VERSIONS])

    bars = ax.bar(x, v, yerr=e, capsize=3,
                  color=[COLORS[vv] for vv in VERSIONS],
                  edgecolor="white", linewidth=0.6,
                  error_kw=dict(elinewidth=0.8, ecolor="#666"))
    ax.set_xticks(x)
    ax.set_xticklabels(VERSIONS, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylim(ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ref1:
        ax.axhline(1.0 if key == "Deadline Pressure" else 0,
                   color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, v):
        fmt = (f"{val:.1f}" if abs(val) >= 1 else f"{val:.3f}")
        ypos = bar.get_height() + max(abs(v)) * 0.01
        if val < 0:
            ypos = val - max(abs(v)) * 0.03
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                fmt, ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8, fontweight="bold")

legend_patches = [plt.Rectangle((0,0),1,1, color=COLORS[v], label=v) for v in VERSIONS]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.savefig(f"{OUTPUT_DIR}/compare_dashboard.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ compare_dashboard.png")

print(f"\n✅ 所有比較圖已輸出至 {OUTPUT_DIR}/")
print("   compare_core_performance.png  — 核心效能 4 合 1")
print("   compare_noma_metrics.png      — NOMA 通道 4 合 1")
print("   compare_delay_decomp.png      — 延遲分解堆疊")
print("   compare_reward_rho.png        — Reward + Rho")
print("   compare_radar.png             — 雷達圖 8 維")
print("   compare_dashboard.png         — 完整 9 合 1 儀表板")
