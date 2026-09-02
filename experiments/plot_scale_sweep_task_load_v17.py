"""
plot_scale_sweep_task_load_v17.py
TD3(V17) scalability vs. task load (MEC=3, CH=3): E2E delay + timeout ratio
across num_tasks in {50, 100, 200, 300, 500}, from results/scale_sweep_v17.csv
(the "num_tasks" axis rows, algo=TD3).

Run: python -m experiments.plot_scale_sweep_task_load_v17
"""

import csv

import matplotlib.pyplot as plt

INPUT_CSV  = "results/scale_sweep_v17.csv"
OUTPUT_PNG = "results/figures_baseline/scale_sweep_v17_task_load.png"

DELAY_COLOR   = "#2E86AB"
TIMEOUT_COLOR = "#E07B54"


def savefig_both(fig, png_path, **kwargs):
    fig.savefig(png_path, dpi=300, **kwargs)
    fig.savefig(png_path.replace(".png", ".svg"), **kwargs)


def main():
    rows = []
    with open(INPUT_CSV) as f:
        for row in csv.DictReader(f):
            if row["axis"] == "num_tasks" and row["algo"] == "TD3":
                rows.append(row)
    rows.sort(key=lambda r: int(r["value"]))

    tasks   = [int(r["num_tasks"]) for r in rows]
    delay   = [float(r["avg_delay"]) for r in rows]
    timeout = [float(r["timeout_ratio"]) * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(tasks, delay, color=DELAY_COLOR, marker="o", markersize=7,
              lw=2, zorder=3)
    ax1.set_xlabel("Tasks per episode")
    ax1.set_ylabel("E2E delay (ms)", color=DELAY_COLOR)
    ax1.tick_params(axis="y", labelcolor=DELAY_COLOR)
    ax1.set_ylim(0, max(delay) * 1.45)
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(tasks, timeout, color=TIMEOUT_COLOR, marker="s", markersize=7,
              lw=2, linestyle="--", zorder=3)
    ax2.set_ylabel("Timeout ratio (%)", color=TIMEOUT_COLOR)
    ax2.tick_params(axis="y", labelcolor=TIMEOUT_COLOR)
    ax2.set_ylim(0, max(timeout) * 1.45)

    ax1.set_title("TD3 Scalability: Task Load (MEC=3, CH=3)", fontsize=12, fontweight="bold")
    fig.tight_layout()

    savefig_both(fig, OUTPUT_PNG)
    print(f"saved {OUTPUT_PNG} (+ .svg)")
    for t, d, to in zip(tasks, delay, timeout):
        print(f"  tasks={t:4d}  delay={d:.3f} ms  timeout={to:.2f}%")


if __name__ == "__main__":
    main()
