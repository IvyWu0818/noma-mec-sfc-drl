import matplotlib.pyplot as plt

# 直接貼你的結果（之後也可以改成自動讀）
data = [
    # scenario, method, delay, slack, obj, timeout
    ("easy", "random", 31.39, 2.81, 59.47, 0.356),
    ("easy", "greedy", 26.66, 1.14, 38.01, 0.208),
    ("easy", "objective", 26.66, 1.14, 38.01, 0.208),

    ("medium", "random", 32.39, 8.29, 115.32, 0.652),
    ("medium", "greedy", 27.66, 5.24, 80.04, 0.524),
    ("medium", "objective", 27.66, 5.24, 80.04, 0.524),

    ("hard", "random", 33.35, 15.33, 186.69, 0.864),
    ("hard", "greedy", 28.29, 10.97, 137.95, 0.764),
    ("hard", "objective", 28.29, 10.97, 137.95, 0.764),
]


def extract(metric_idx):
    scenarios = ["easy", "medium", "hard"]
    methods = ["random", "greedy", "objective"]

    result = {m: [] for m in methods}

    for s in scenarios:
        for m in methods:
            for row in data:
                if row[0] == s and row[1] == m:
                    result[m].append(row[metric_idx])
    return scenarios, result


def plot_metric(metric_idx, title, ylabel):
    scenarios, result = extract(metric_idx)

    x = range(len(scenarios))

    plt.figure()

    for m in result:
        plt.plot(x, result[m], marker='o', label=m)

    plt.xticks(x, scenarios)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.show()


def main():
    plot_metric(2, "Avg Delay", "Delay")
    plot_metric(3, "Avg Slack", "Slack")
    plot_metric(5, "Timeout Ratio", "Ratio")
    plot_metric(4, "Avg Objective", "Objective")


if __name__ == "__main__":
    main()