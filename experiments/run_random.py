import random
from core.task import Task
from core.sfc import VNF, SFC
from core.mec import MECNode
from core.topology import create_topology
from core.delay import total_delay


def create_random_task(task_id):
    vnfs = [VNF(i, random.randint(5, 15)) for i in range(3)]
    sfc = SFC(vnfs)

    return Task(
        task_id=task_id,
        data_size=random.randint(10, 30),
        deadline=random.randint(20, 50),
        sfc_chain=sfc
    )


def run():
    graph = create_topology()

    mec_nodes = {
        "mec0": MECNode("mec0", 50),
        "mec1": MECNode("mec1", 50),
        "mec2": MECNode("mec2", 50)
    }

    tasks = [create_random_task(i) for i in range(10)]

    for task in tasks:
        for _ in task.sfc_chain.vnfs:
            node = random.choice(list(mec_nodes.keys()))
            cpu = random.uniform(5, 20)

            task.vnf_placement.append(node)
            task.cpu_alloc.append(cpu)

        task.total_delay = total_delay(task, graph, mec_nodes)

        print(f"Task {task.task_id}")
        print(f"  Delay: {task.total_delay:.2f}")
        print(f"  Deadline: {task.deadline}")
        print(f"  Timeout: {task.total_delay > task.deadline}")
        print("-" * 30)


if __name__ == "__main__":
    run()