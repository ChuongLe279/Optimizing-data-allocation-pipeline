import networkx as nx
import numpy as np
from scipy.sparse import csr_array


"""Class cốt lõi"""
class PartitionScheme():
    def __init__(self, heat_graph: nx.Graph, partitions: dict[int, set[int]]):
        self.heat_graph = heat_graph
        self.partitions = partitions

class BregmanProximalGradient(): # BPG
    def __init__(self, W: csr_array, partitions_num: int, step_size: int, iterations_num: int):
        self.W = W
        self.partitions_num = partitions_num
        self.step_size = step_size
        self.iterations_num = iterations_num
    
    def run(self) -> np.ndarray:
        return np.array([])

class PartitioningUnbalancedGraph(): # Algorithm 3
    def __init__(self, heat_graph: nx.Graph, partition_scheme: PartitionScheme, site_num: int):
        self.heat_graph = heat_graph
        self.partition_scheme = partition_scheme
        self.site_num = site_num

    def run(self):
        return self.heat_graph

class PartitioningMinimumCutGraph(): # Algorithm 4
    def __init__(self, heat_graph: nx.Graph, partition_scheme: PartitionScheme, site_num: int, depth_search: int):
        self.heat_graph = heat_graph
        self.partition_scheme = partition_scheme
        self.site_num = site_num
        self.depth_search = depth_search

    def run(self):
        return self.heat_graph

class FirstPartitioning(): # Algorithm 5
    def __init__(self, heat_graph: nx.Graph, partition_scheme: PartitionScheme, site_num: int):
        self.heat_graph = heat_graph
        self.partition_scheme = partition_scheme
        self.site_num = site_num

    def run(self):
        return self.heat_graph

class Partitioning(): # Algorithm 2
    def __init__(self, heat_graph: nx.Graph, site_num: int, workload_factor: float, distributed_percentage: float):
        self.heat_graph = heat_graph
        self.site_num = site_num
        self.workload_factor = workload_factor
        self.distributed_percentage = distributed_percentage

    def run(self):
        return self.heat_graph

"""Class phụ trợ"""
class CreatePartitionScheme():
    """Dữ liệu giả, sau này sẽ chuyển thành hàm Database -> Partitioning Scheme"""
    def __init__(self):
        pass
    
    def run(self):
        pass

class BPG_PartitionScheme():
    def __init__(self, labels: np.ndarray):
        self.labels = labels
    
    def run(self):
        pass

class PartitionScheme_BPG():
    def __init__(self, partition_scheme: PartitionScheme):
        pass
    
    def run(self):
        pass




