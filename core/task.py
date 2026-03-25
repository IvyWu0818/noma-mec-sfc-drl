class Task:
    def __init__(self, task_id, data_size, deadline, sfc_chain):
        self.task_id = task_id
        self.data_size = data_size
        self.deadline = deadline
        self.sfc_chain = sfc_chain

        self.vnf_placement = []
        self.cpu_alloc = []
        self.total_delay = 0.0