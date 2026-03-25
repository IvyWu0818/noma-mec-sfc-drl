class MECNode:
    def __init__(self, node_id, cpu_capacity):
        self.node_id = node_id
        self.cpu_capacity = cpu_capacity
        self.available_cpu = cpu_capacity
        self.queue = []

    def estimate_waiting_time(self):
        return sum(self.queue)