import random
from statistics import mean

from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay, local_stage_delay

random.seed(42)


def create_random_task(task_id):
    vnfs = [VNF(i, random.randint(10, 25)) for i in range(3)]
    sfc = SFC(vnfs)

    return Task(
        task_id=task_id,
        data_size=random.randint(20, 50),
        deadline=random.randint(25, 45),
        sfc_chain=sfc
    )


def create_mec_nodes():
    return {
        "mec0": MECNode("mec0", 50),
        "mec1": MECNode("mec1", 50),
        "mec2": MECNode("mec2", 50)
    }


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


def run():
    graph = create_topology()
    mec_nodes = create_mec_nodes()
    tasks = [create_random_task(i) for i in range(10)]

    delays = []
    timeouts = 0

    for task in tasks:
        greedy_assign_task(task, graph, mec_nodes)
        task.total_delay = total_delay(task, graph, mec_nodes)
        delays.append(task.total_delay)

        timeout = task.total_delay > task.deadline
        if timeout:
            timeouts += 1

        print(f"Task {task.task_id}")
        print(f"  Placement: {task.vnf_placement}")
        print(f"  CPU alloc: {[round(x, 2) for x in task.cpu_alloc]}")
        print(f"  Delay: {task.total_delay:.2f}")
        print(f"  Deadline: {task.deadline}")
        print(f"  Timeout: {timeout}")
        print("-" * 30)

    print("Summary")
    print(f"  Avg delay: {mean(delays):.2f}")
    print(f"  Timeout ratio: {timeouts / len(tasks):.2%}")


if __name__ == "__main__":
    run()