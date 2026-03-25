class VNF:
    def __init__(self, vnf_id, cpu_cycles):
        self.vnf_id = vnf_id
        self.cpu_cycles = cpu_cycles


class SFC:
    def __init__(self, vnfs):
        self.vnfs = vnfs