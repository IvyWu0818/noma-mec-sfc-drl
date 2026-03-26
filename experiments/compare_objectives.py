import copy
import random
from statistics import mean
from typing import Dict, List

from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay, local_stage_delay
from core.objective import compute_slack, compute_objective


RANDOM_SEED = 42


# ============================================================
# 基本資料生成
# ============================================================

def create_mec_nodes() -> Dict[str, MECNode]:
    return {
        "mec0": MECNode("mec0", 50),
        "mec1": MECNode("mec1", 50),
        "mec2": MECNode("mec2", 50),
    }


def create_task(
    task_id: int,
    data_range: tuple,
    deadline_range: tuple,
    num_vnfs: int = 3,
    cycle_range: tuple = (10, 25),
) -> Task:
    vnfs = [VNF(i, random.randint(cycle_range[0], cycle_range[1])) for i in range(num_vnfs)]
    sfc = SFC(vnfs)

    return Task(
        task_id=task_id,
        data_size=random.randint(data_range[0], data_range[1]),
        deadline=random.randint(deadline_range[0], deadline_range[1]),
        sfc_chain=sfc,
    )


def generate_tasks(
    num_tasks: int,
    data_range: tuple,
    deadline_range: tuple,
    num_vnfs: int = 3,
    cycle_range: tuple = (10, 25),
) -> List[Task]:
    return [
        create_task(
            task_id=i,
            data_range=data_range,
            deadline_range=deadline_range,
            num_vnfs=num_vnfs,
            cycle_range=cycle_range,
        )
        for i in range(num_tasks)
    ]


def clone_tasks(tasks: List[Task]) -> List[Task]:
    """
    確保三種方法用的是完全相同的一批 task。
    """
    return copy.deepcopy(tasks)


# ============================================================
# 三種方法
# ============================================================

def random_assign_task(task: Task, mec_nodes: Dict[str, MECNode]) -> None:
    """
    random baseline:
    每段 VNF 隨機選一個節點，CPU 也隨機給一個範圍。
    """
    node_ids = list(mec_nodes.keys())

    for vnf in task.sfc_chain.vnfs:
        node_id = random.choice(node_ids)
        cpu_alloc = random.uniform(8.0, 20.0)

        task.vnf_placement.append(node_id)
        task.cpu_alloc.append(cpu_alloc)

        service_time = vnf.cpu_cycles / cpu_alloc
        mec_nodes[node_id].queue_load += service_time * 0.2


def greedy_assign_task(task: Task, graph, mec_nodes: Dict[str, MECNode]) -> None:
    """
    delay-only greedy:
    每段 VNF 都選 stage delay 最小的節點。
    """
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
                mec_nodes=mec_nodes,
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


def objective_greedy_assign_task(
    task: Task,
    graph,
    mec_nodes: Dict[str, MECNode],
    beta: float = 10.0,
) -> None:
    """
    objective greedy:
    逐段選擇 cost 最小者，其中
        cost = estimated_total_delay + beta * slack
    """
    estimated_total = task.data_size / 10.0  # 對應 uplink_delay 預設 rate=10
    prev_node = None

    for vnf in task.sfc_chain.vnfs:
        best_node = None
        best_cpu = None
        best_cost = float("inf")
        best_stage_delay = None

        for node_id, node in mec_nodes.items():
            candidate_cpu = min(max(vnf.cpu_cycles * 0.8, 8.0), node.cpu_capacity * 0.6)

            stage_delay = local_stage_delay(
                graph=graph,
                prev_node=prev_node,
                node_id=node_id,
                vnf=vnf,
                cpu_alloc=candidate_cpu,
                mec_nodes=mec_nodes,
            )

            candidate_total = estimated_total + stage_delay
            candidate_slack = compute_slack(candidate_total, task.deadline)
            candidate_cost = candidate_total + beta * candidate_slack

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_node = node_id
                best_cpu = candidate_cpu
                best_stage_delay = stage_delay

        task.vnf_placement.append(best_node)
        task.cpu_alloc.append(best_cpu)

        service_time = vnf.cpu_cycles / best_cpu
        mec_nodes[best_node].queue_load += service_time * 0.2

        estimated_total += best_stage_delay
        prev_node = best_node


# ============================================================
# 評估
# ============================================================

def evaluate_method(
    tasks: List[Task],
    method: str,
    beta: float = 10.0,
) -> Dict[str, float]:
    graph = create_topology()
    mec_nodes = create_mec_nodes()

    delays = []
    slacks = []
    objectives = []
    timeouts = 0

    for task in tasks:
        if method == "random":
            random_assign_task(task, mec_nodes)
        elif method == "greedy_delay":
            greedy_assign_task(task, graph, mec_nodes)
        elif method == "objective_greedy":
            objective_greedy_assign_task(task, graph, mec_nodes, beta=beta)
        else:
            raise ValueError(f"Unknown method: {method}")

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


# ============================================================
# 單一 scenario 比較
# ============================================================

def compare_methods_once(
    scenario_name: str,
    num_tasks: int,
    data_range: tuple,
    deadline_range: tuple,
    num_vnfs: int,
    cycle_range: tuple,
    beta: float,
    seed: int,
) -> List[Dict[str, float]]:
    random.seed(seed)

    base_tasks = generate_tasks(
        num_tasks=num_tasks,
        data_range=data_range,
        deadline_range=deadline_range,
        num_vnfs=num_vnfs,
        cycle_range=cycle_range,
    )

    methods = ["random", "greedy_delay", "objective_greedy"]
    rows = []

    for method in methods:
        tasks = clone_tasks(base_tasks)
        result = evaluate_method(tasks, method=method, beta=beta)

        row = {
            "scenario": scenario_name,
            "method": method,
            **result,
        }
        rows.append(row)

    return rows


# ============================================================
# 多次重複平均
# ============================================================

def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "avg_delay": mean(r["avg_delay"] for r in rows),
        "avg_slack": mean(r["avg_slack"] for r in rows),
        "avg_objective": mean(r["avg_objective"] for r in rows),
        "timeout_ratio": mean(r["timeout_ratio"] for r in rows),
        "max_delay": mean(r["max_delay"] for r in rows),
        "max_slack": mean(r["max_slack"] for r in rows),
    }


def compare_methods_across_scenarios(
    beta: float = 10.0,
    repeats: int = 5,
) -> List[Dict[str, float]]:
    scenarios = {
        "easy": {
            "num_tasks": 50,
            "data_range": (20, 40),
            "deadline_range": (30, 45),
            "num_vnfs": 3,
            "cycle_range": (10, 25),
        },
        "medium": {
            "num_tasks": 50,
            "data_range": (30, 50),
            "deadline_range": (20, 35),
            "num_vnfs": 3,
            "cycle_range": (10, 25),
        },
        "hard": {
            "num_tasks": 50,
            "data_range": (35, 60),
            "deadline_range": (12, 25),
            "num_vnfs": 3,
            "cycle_range": (10, 25),
        },
    }

    collected = {}

    for rep in range(repeats):
        seed = RANDOM_SEED + rep

        for scenario_name, cfg in scenarios.items():
            rows = compare_methods_once(
                scenario_name=scenario_name,
                num_tasks=cfg["num_tasks"],
                data_range=cfg["data_range"],
                deadline_range=cfg["deadline_range"],
                num_vnfs=cfg["num_vnfs"],
                cycle_range=cfg["cycle_range"],
                beta=beta,
                seed=seed,
            )

            for row in rows:
                key = (row["scenario"], row["method"])
                collected.setdefault(key, []).append(row)

    final_rows = []
    for (scenario, method), rows in collected.items():
        agg = aggregate_rows(rows)
        final_rows.append({
            "scenario": scenario,
            "method": method,
            **agg,
        })

    # 排序輸出：easy -> medium -> hard，再按 method
    scenario_order = {"easy": 0, "medium": 1, "hard": 2}
    method_order = {"random": 0, "greedy_delay": 1, "objective_greedy": 2}

    final_rows.sort(key=lambda x: (scenario_order[x["scenario"]], method_order[x["method"]]))
    return final_rows


# ============================================================
# 顯示結果
# ============================================================

def print_results_table(rows: List[Dict[str, float]], beta: float, repeats: int) -> None:
    print("=" * 120)
    print("Compare objectives under the same scenarios")
    print("=" * 120)
    print(f"Beta = {beta}")
    print(f"Repeats = {repeats}")
    print()

    print(
        f"{'Scenario':<10}"
        f"{'Method':<18}"
        f"{'AvgDelay':<12}"
        f"{'AvgSlack':<12}"
        f"{'AvgObj':<12}"
        f"{'TimeoutRatio':<14}"
        f"{'MaxDelay':<12}"
        f"{'MaxSlack':<12}"
    )

    for row in rows:
        print(
            f"{row['scenario']:<10}"
            f"{row['method']:<18}"
            f"{row['avg_delay']:<12.2f}"
            f"{row['avg_slack']:<12.2f}"
            f"{row['avg_objective']:<12.2f}"
            f"{row['timeout_ratio']:<14.2%}"
            f"{row['max_delay']:<12.2f}"
            f"{row['max_slack']:<12.2f}"
        )

    print()
    print("How to interpret:")
    print("- random: weak baseline")
    print("- greedy_delay: minimizes delay stage by stage")
    print("- objective_greedy: minimizes delay + beta * slack stage by stage")
    print("- If objective_greedy reduces timeout/slack, it supports the usefulness of Formula (1).")


def main():
    beta = 10.0
    repeats = 5

    rows = compare_methods_across_scenarios(beta=beta, repeats=repeats)
    print_results_table(rows, beta=beta, repeats=repeats)


if __name__ == "__main__":
    main()