import random
from statistics import mean

from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay, local_stage_delay
from core.objective import compute_slack, compute_objective

random.seed(42)


def create_mec_nodes():
    return {
        "mec0": MECNode("mec0", 50),
        "mec1": MECNode("mec1", 50),
        "mec2": MECNode("mec2", 50)
    }


def create_task(task_id, data_range, deadline_range, num_vnfs=3, cycle_range=(10, 25)):
    vnfs = [VNF(i, random.randint(cycle_range[0], cycle_range[1])) for i in range(num_vnfs)]
    sfc = SFC(vnfs)

    return Task(
        task_id=task_id,
        data_size=random.randint(data_range[0], data_range[1]),
        deadline=random.randint(deadline_range[0], deadline_range[1]),
        sfc_chain=sfc
    )


def generate_tasks(num_tasks, data_range, deadline_range):
    return [create_task(i, data_range, deadline_range) for i in range(num_tasks)]


def greedy_assign_task(task, graph, mec_nodes):
    prev_node = None

    for vnf in task.sfc_chain.vnfs:
        best_node = None
        best_cpu = None
        best_cost = float("inf")

        for node_id, node in mec_nodes.items():
            candidate_cpu = min(max(vnf.cpu_cycles * 0.8, 8.0), node.cpu_capacity * 0.6)

            stage_cost = local_stage_delay(
                graph=graph,
                prev_node=prev_node,
                node_id=node_id,
                vnf=vnf,
                cpu_alloc=candidate_cpu,
                mec_nodes=mec_nodes
            )

            if stage_cost < best_cost:
                best_cost = stage_cost
                best_node = node_id
                best_cpu = candidate_cpu

        task.vnf_placement.append(best_node)
        task.cpu_alloc.append(best_cpu)

        service_time = vnf.cpu_cycles / best_cpu
        mec_nodes[best_node].queue_load += service_time * 0.2

        prev_node = best_node


def evaluate_tasks(tasks, beta):
    graph = create_topology()
    mec_nodes = create_mec_nodes()

    delays = []
    slacks = []
    objectives = []
    timeouts = 0

    for task in tasks:
        greedy_assign_task(task, graph, mec_nodes)
        delay = total_delay(task, graph, mec_nodes)
        slack = compute_slack(delay, task.deadline)
        objective = compute_objective(delay, task.deadline, beta)

        task.total_delay = delay

        delays.append(delay)
        slacks.append(slack)
        objectives.append(objective)

        if delay > task.deadline:
            timeouts += 1

    return {
        "avg_delay": mean(delays),
        "avg_slack": mean(slacks),
        "avg_objective": mean(objectives),
        "timeout_ratio": timeouts / len(tasks),
        "max_delay": max(delays),
        "max_slack": max(slacks),
    }


def test_formula_78_basic_cases():
    print("=" * 60)
    print("Test A: Validate formula (7)(8) slack definition")
    print("=" * 60)

    cases = [
        {"delay": 10.0, "deadline": 15.0},
        {"delay": 15.0, "deadline": 15.0},
        {"delay": 20.0, "deadline": 15.0},
    ]

    print(f"{'Case':<8}{'Delay':<10}{'Deadline':<12}{'Slack':<10}")
    for idx, case in enumerate(cases, start=1):
        slack = compute_slack(case["delay"], case["deadline"])
        print(f"{idx:<8}{case['delay']:<10.2f}{case['deadline']:<12.2f}{slack:<10.2f}")

    print()
    print("Expected behavior:")
    print("- delay < deadline  -> slack = 0")
    print("- delay = deadline  -> slack = 0")
    print("- delay > deadline  -> slack = delay - deadline")
    print()


def test_formula_1_beta_sensitivity():
    print("=" * 60)
    print("Test B: Validate formula (1) objective with different beta")
    print("=" * 60)

    # Use one fixed scenario so beta is the only changing factor
    random.seed(42)
    tasks_template = generate_tasks(
        num_tasks=30,
        data_range=(30, 50),
        deadline_range=(20, 35),
    )

    betas = [0, 1, 5, 10, 20]

    print(f"{'Beta':<8}{'AvgDelay':<12}{'AvgSlack':<12}{'AvgObj':<12}{'TimeoutRatio':<14}")
    for beta in betas:
        # regenerate the same tasks for fair comparison
        random.seed(42)
        tasks = generate_tasks(
            num_tasks=30,
            data_range=(30, 50),
            deadline_range=(20, 35),
        )

        result = evaluate_tasks(tasks, beta)

        print(
            f"{beta:<8}"
            f"{result['avg_delay']:<12.2f}"
            f"{result['avg_slack']:<12.2f}"
            f"{result['avg_objective']:<12.2f}"
            f"{result['timeout_ratio']:<14.2%}"
        )

    print()
    print("Interpretation:")
    print("- AvgDelay reflects the original delay cost.")
    print("- AvgSlack reflects deadline violation severity.")
    print("- AvgObj = delay + beta * slack.")
    print("- When beta increases, the objective should penalize timeout more strongly.")
    print()


def test_under_different_load_levels():
    print("=" * 60)
    print("Test C: Validate formulas under easy / medium / hard scenarios")
    print("=" * 60)

    scenarios = {
        "easy": {
            "data_range": (20, 40),
            "deadline_range": (30, 45),
        },
        "medium": {
            "data_range": (30, 50),
            "deadline_range": (20, 35),
        },
        "hard": {
            "data_range": (35, 60),
            "deadline_range": (12, 25),
        },
    }

    beta = 10

    print(f"{'Scenario':<10}{'AvgDelay':<12}{'AvgSlack':<12}{'AvgObj':<12}{'TimeoutRatio':<14}{'MaxDelay':<12}")
    for scenario_name, cfg in scenarios.items():
        random.seed(42)
        tasks = generate_tasks(
            num_tasks=50,
            data_range=cfg["data_range"],
            deadline_range=cfg["deadline_range"],
        )

        result = evaluate_tasks(tasks, beta)

        print(
            f"{scenario_name:<10}"
            f"{result['avg_delay']:<12.2f}"
            f"{result['avg_slack']:<12.2f}"
            f"{result['avg_objective']:<12.2f}"
            f"{result['timeout_ratio']:<14.2%}"
            f"{result['max_delay']:<12.2f}"
        )

    print()
    print("Expected trend:")
    print("- easy   -> lower delay / lower slack / lower timeout ratio")
    print("- medium -> moderate values")
    print("- hard   -> higher delay / higher slack / higher timeout ratio")
    print()


def main():
    test_formula_78_basic_cases()
    test_formula_1_beta_sensitivity()
    test_under_different_load_levels()


if __name__ == "__main__":
    main()