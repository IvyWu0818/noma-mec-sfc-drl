import random
from statistics import mean

from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay

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


def run():
    graph = create_topology()
    mec_nodes = create_mec_nodes()
    tasks = [create_random_task(i) for i in range(10)]

    delays = []
    timeouts = 0

    for task in tasks:
        for vnf in task.sfc_chain.vnfs:
            node_id = random.choice(list(mec_nodes.keys()))
            cpu_alloc = random.uniform(8, 20)

            task.vnf_placement.append(node_id)
            task.cpu_alloc.append(cpu_alloc)

            service_time = vnf.cpu_cycles / cpu_alloc
            mec_nodes[node_id].queue_load += service_time * 0.2

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