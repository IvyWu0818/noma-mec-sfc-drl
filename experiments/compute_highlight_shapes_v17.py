"""
compute_highlight_shapes_v17.py
For each of the 11 v17 deck charts, builds the *original* (unmodified)
chart exactly as its source script does, then works out where the
highlight ellipse/box would sit as a fraction (0..1, top-left origin) of
the saved image -- using ax.transData directly rather than adding a
matplotlib patch, so this never risks nudging the original chart's layout.

Output: results/figures_highlight/shapes.json, one entry per chart
filename (matching the PNG already embedded in the deck), each with a
shape type, its bbox as image fractions, and (for the two line charts
that have no printed data label) a short callout string.

Run: python -m experiments.compute_highlight_shapes_v17
"""

import os
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_JSON = "results/figures_highlight/shapes.json"

ALGOS = ["TD3", "Greedy", "GA"]
COLORS = {"TD3": "#2E86AB", "Greedy": "#E07B54", "GA": "#7E57C2"}


def mean_std(data, algo, key):
    vals = data.get(algo, {}).get(key, [])
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (0.0, 0.0)


def frac_bbox_from_data(ax, x0, y0, x1, y1):
    """Convert a data-space rectangle to an image-fraction bbox (top-left origin)."""
    fig = ax.figure
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    xs = [p0[0], p1[0]]
    ys = [p0[1], p1[1]]
    fx0, fx1 = min(xs) / w, max(xs) / w
    # display y is bottom-up; flip to top-down image fraction
    fy0, fy1 = 1 - max(ys) / h, 1 - min(ys) / h
    return {"x0": fx0, "y0": fy0, "x1": fx1, "y1": fy1}


def ellipse_bbox(ax, cx, cy, rx, ry):
    return frac_bbox_from_data(ax, cx - rx, cy - ry, cx + rx, cy + ry)


def build_bar_panel(ax, data, key, title, ylabel):
    means = [mean_std(data, a, key)[0] for a in ALGOS]
    stds = [mean_std(data, a, key)[1] for a in ALGOS]
    bars = ax.bar(ALGOS, means, yerr=stds, capsize=4,
                   color=[COLORS[a] for a in ALGOS], alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    return bars, means


def shape_for_baseline_bar(key, title, ylabel, out_name, highlight_algo="TD3", log=False):
    with open("results/baseline_eval_v17.json") as f:
        data = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    if log:
        means, stds = [], []
        for a in ALGOS:
            vals = np.array(data.get(a, {}).get("episode_runtime_sec", [])) * 1000.0
            means.append(float(vals.mean()) if len(vals) else 0.0)
            stds.append(float(vals.std()) if len(vals) else 0.0)
        bars = ax.bar(ALGOS, means, yerr=stds, capsize=4,
                       color=[COLORS[a] for a in ALGOS], alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, which="both")
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    else:
        bars, means = build_bar_panel(ax, data, key, title, ylabel)
    fig.tight_layout()

    idx = ALGOS.index(highlight_algo)
    bar = bars[idx]
    cx = bar.get_x() + bar.get_width() / 2
    h = bar.get_height()
    if log:
        cy = h * 1.12
        bbox = ellipse_bbox(ax, cx, np.log10(cy) if False else cy, bar.get_width() * 0.325, h * 0.45)
        # log-scale y: transData handles the log mapping itself, so pass
        # linear cy +/- linear ry is wrong -- do it in two transData calls instead
        bbox = frac_bbox_from_data(ax, cx - bar.get_width() * 0.325, h * 0.67,
                                    cx + bar.get_width() * 0.325, h * 1.57)
    else:
        ylo, yhi = ax.get_ylim()
        span = yhi - ylo
        ew = bar.get_width() * 0.55 + 2 * 0.06
        eh = span * 0.14
        bbox = ellipse_bbox(ax, cx, h, ew / 2, eh / 2)
    plt.close(fig)
    return out_name, {"shape": "ellipse", "bbox": bbox}


def shape_for_decomp():
    with open("results/baseline_eval_v17.json") as f:
        data = json.load(f)
    DECOMP_KEYS = ["episode_avg_t_ul", "episode_avg_t_comp", "episode_avg_t_link"]
    DECOMP_COLORS = ["#7DC3E8", "#F0A070", "#82C882"]
    DECOMP_LABELS = ["t_ul (upload)", "t_comp (compute)", "t_link (backhaul)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(ALGOS))
    for key, label, color in zip(DECOMP_KEYS, DECOMP_LABELS, DECOMP_COLORS):
        means = np.array([mean_std(data, a, key)[0] for a in ALGOS])
        ax.bar(ALGOS, means, bottom=bottoms, color=color, alpha=0.85, label=label)
        bottoms += means
    ax.set_title("Delay Decomposition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Delay (ms)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    total_td3 = bottoms[0]
    bbox = frac_bbox_from_data(ax, -0.4, 0, 0.4, total_td3)
    plt.close(fig)
    return "baseline_compare_v17_delay_decomp.png", {"shape": "roundRect", "bbox": bbox}


def shape_for_training_convergence():
    with open("results/td3_v17_training_metrics.json") as f:
        data = json.load(f)

    def smooth(vals, window=15):
        out = []
        for i in range(len(vals)):
            lo = max(0, i - window + 1)
            out.append(sum(vals[lo:i + 1]) / (i - lo + 1))
        return out

    eps = list(range(len(data["episode_rewards"])))
    smoothed = smooth(data["episode_rewards"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eps, data["episode_rewards"], color="lightgray", alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(eps, smoothed, color="steelblue", linewidth=2.0, label="smoothed (w=15)")
    ax.set_title("V17 Training -- Cumulative Reward", fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    plateau_y = float(np.mean(smoothed[-200:]))
    x_lo = eps[-1] * 0.58
    ylo, yhi = ax.get_ylim()
    bbox = frac_bbox_from_data(ax, x_lo, plateau_y - (yhi - ylo) * 0.075,
                                eps[-1] * 1.02, plateau_y + (yhi - ylo) * 0.075)
    plt.close(fig)
    return "v17_reward.png", {
        "shape": "ellipse", "bbox": bbox,
        "label": "converged eval reward ≈ -12.407",
    }


def shape_for_ga_convergence():
    with open("results/ga_convergence_v17.json") as f:
        data = json.load(f)
    mean = np.array(data["fitness_mean_per_gen"])
    gens = np.arange(1, len(mean) + 1)
    checkpoints = data["checkpoint_summary"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, mean, color="#7E57C2", lw=2, zorder=3)
    ax.axvline(15, color="#E07B54", linestyle="--", lw=1.6, zorder=1)
    last_g, last_v = None, None
    for g_str, ck in sorted(checkpoints.items(), key=lambda kv: int(kv[0])):
        g = int(g_str)
        v = mean[g - 1]
        ax.scatter([g], [v], color="#7E57C2", s=28, zorder=4)
        last_g, last_v = g, v
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (cumulative reward)")
    ax.set_title("GA(V17) Convergence (300 Generations)")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    bbox = ellipse_bbox(ax, last_g, last_v, (xhi - xlo) * 0.04, (yhi - ylo) * 0.075)
    plt.close(fig)
    return "ga_convergence_v17_delay_annotated.png", {"shape": "ellipse", "bbox": bbox}


def shape_for_task_load():
    rows = []
    with open("results/scale_sweep_v17.csv") as f:
        for row in csv.DictReader(f):
            if row["axis"] == "num_tasks" and row["algo"] == "TD3":
                rows.append(row)
    rows.sort(key=lambda r: int(r["value"]))
    tasks = [int(r["num_tasks"]) for r in rows]
    delay = [float(r["avg_delay"]) for r in rows]
    timeout = [float(r["timeout_ratio"]) * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(tasks, delay, color="#2E86AB", marker="o", markersize=7, lw=2, zorder=3)
    ax1.set_ylim(0, max(delay) * 1.45)
    ax2 = ax1.twinx()
    ax2.plot(tasks, timeout, color="#E07B54", marker="s", markersize=7, lw=2, linestyle="--", zorder=3)
    ax2.set_ylim(0, max(timeout) * 1.45)
    ax1.set_title("TD3 Scalability: Task Load (MEC=3, CH=3)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    x500, y500 = tasks[-1], delay[-1]
    ylo, yhi = ax1.get_ylim()
    xlo, xhi = ax1.get_xlim()
    bbox = ellipse_bbox(ax1, x500, y500, (xhi - xlo) * 0.035, (yhi - ylo) * 0.09)
    plt.close(fig)
    return "scale_sweep_v17_task_load.png", {
        "shape": "ellipse", "bbox": bbox,
        "label": "7.004 ms at 500 tasks (below the 100-task baseline)",
    }


def shape_for_system_size():
    CONFIGS = [
        ("3 MEC/3 CH", "results/td3_v17_tasks100_mec3_ch3_training_metrics.json"),
        ("5 MEC/3 CH", "results/td3_v17_tasks100_mec5_ch3_training_metrics.json"),
        ("7 MEC/3 CH", "results/td3_v17_tasks100_mec7_ch3_training_metrics.json"),
        ("3 MEC/5 CH", "results/td3_v17_tasks100_mec3_ch5_training_metrics.json"),
    ]
    TAIL_FRAC = 0.10

    def tail_mean(vals, frac):
        n = max(1, int(len(vals) * frac))
        return float(np.mean(vals[-n:]))

    labels, t_ul, t_comp, t_link = [], [], [], []
    for label, path in CONFIGS:
        with open(path) as f:
            d = json.load(f)
        labels.append(label)
        t_ul.append(tail_mean(d["episode_avg_t_ul"], TAIL_FRAC))
        t_comp.append(tail_mean(d["episode_avg_t_comp"], TAIL_FRAC))
        t_link.append(tail_mean(d["episode_avg_t_link"], TAIL_FRAC))
    t_ul, t_comp, t_link = map(np.array, (t_ul, t_comp, t_link))
    totals = t_ul + t_comp + t_link

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(x, t_ul, color="#7dc3e8")
    ax.bar(x, t_comp, bottom=t_ul, color="#f0a070")
    ax.bar(x, t_link, bottom=t_ul + t_comp, color="#82c882")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delay (ms)")
    ax.set_title("Delay Decomposition vs. System Size (100 tasks)")
    ax.set_ylim(0, totals.max() * 1.15)
    fig.tight_layout()

    idx = 2  # "7 MEC/3 CH"
    bbox = ellipse_bbox(ax, x[idx], totals[idx], 0.4, (totals.max() * 1.15) * 0.07)
    plt.close(fig)
    return "scale_sweep_v17_system_size_decomp.png", {"shape": "ellipse", "bbox": bbox}


def main():
    results = {}
    results.update([shape_for_training_convergence()])
    results.update([shape_for_baseline_bar("episode_rewards", "Cumulative Reward", "Reward",
                                            "baseline_compare_v17_reward.png")])
    results.update([shape_for_ga_convergence()])
    results.update([shape_for_baseline_bar("episode_avg_delay", "End-to-End Delay (ms)", "Avg Delay (ms)",
                                            "baseline_compare_v17_delay.png")])
    results.update([shape_for_decomp()])
    results.update([shape_for_baseline_bar("episode_cpu_viol_rate", "CPU Violation Rate", "Rate (raw / 135)",
                                            "baseline_compare_v17_cpu_viol.png")])
    results.update([shape_for_baseline_bar("episode_channel_overflow_ratio", "Channel Overflow Ratio", "Ratio",
                                            "baseline_compare_v17_ch_overflow.png")])
    results.update([shape_for_baseline_bar("episode_timeout_ratio", "Timeout Ratio", "Ratio",
                                            "baseline_compare_v17_timeout.png")])
    results.update([shape_for_baseline_bar(None, "Per-Episode Computation Time", "Runtime (ms, log scale)",
                                            "baseline_compare_v17_runtime.png", log=True)])
    results.update([shape_for_task_load()])
    results.update([shape_for_system_size()])

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {OUT_JSON}")
    for k, v in results.items():
        b = v["bbox"]
        print(f"  {k:50s} {v['shape']:10s} x[{b['x0']:.3f},{b['x1']:.3f}] y[{b['y0']:.3f},{b['y1']:.3f}]"
              + (f"  label={v['label']!r}" if "label" in v else ""))


if __name__ == "__main__":
    main()
