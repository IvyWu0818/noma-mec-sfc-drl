import random
from statistics import mean

from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay, local_stage_delay
from core.objective import compute_slack, compute_objective

random.seed(42)


def create_random_task(task_id):
    vnfs = [VNF(i, random.randint(10, 30)) for i in range(4)]
    sfc = SFC(vnfs)

    return Task(
        task_id=task_id,
        deadline=random.randint(12, 25),
        data_size=random.randint(40, 70),
        sfc_chain=sfc
    )


def create_mec_nodes():
    return {
        "mec0": MECNode("mec0", 50),
        "mec1": MECNode("mec1", 50),
        "mec2": MECNode("mec2", 50)
    }


def objective_greedy_assign_task(task, graph, mec_nodes, beta=10.0):
    """
    Choose placement stage by stage using:
        cost = estimated_total_delay_so_far + beta * slack

    where:
        slack = max(0, estimated_total_delay_so_far - deadline)
    """
    estimated_total = task.data_size / 10.0  # same as uplink_delay default
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
                mec_nodes=mec_nodes
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


def run(beta=10.0, num_tasks=10):
    graph = create_topology()
    mec_nodes = create_mec_nodes()
    tasks = [create_random_task(i) for i in range(num_tasks)]

    delays = []
    slacks = []
    objectives = []
    timeouts = 0

    for task in tasks:
        objective_greedy_assign_task(task, graph, mec_nodes, beta=beta)

        task.total_delay = total_delay(task, graph, mec_nodes)
        slack = compute_slack(task.total_delay, task.deadline)
        objective = compute_objective(task.total_delay, task.deadline, beta)

        delays.append(task.total_delay)
        slacks.append(slack)
        objectives.append(objective)

        timeout = task.total_delay > task.deadline
        if timeout:
            timeouts += 1

        print(f"Task {task.task_id}")
        print(f"  Placement: {task.vnf_placement}")
        print(f"  CPU alloc: {[round(x, 2) for x in task.cpu_alloc]}")
        print(f"  Delay: {task.total_delay:.2f}")
        print(f"  Deadline: {task.deadline}")
        print(f"  Slack: {slack:.2f}")
        print(f"  Objective: {objective:.2f}")
        print(f"  Timeout: {timeout}")
        print("-" * 30)

    print("Summary")
    print(f"  Beta: {beta}")
    print(f"  Avg delay: {mean(delays):.2f}")
    print(f"  Avg slack: {mean(slacks):.2f}")
    print(f"  Avg objective: {mean(objectives):.2f}")
    print(f"  Timeout ratio: {timeouts / len(tasks):.2%}")
    print(f"  Max delay: {max(delays):.2f}")
    print(f"  Max slack: {max(slacks):.2f}")


if __name__ == "__main__":
    run(beta=10.0, num_tasks=10)