import networkx as nx
import numpy as np
from scipy.sparse import csr_array
import copy

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
        rng = np.random.default_rng()
        N = self.W.shape[0] # type: ignore
        K = self.partitions_num
        X = (1 / K) * np.ones((N, K)) + 0.1 * rng.random((N,K))

        # Chuẩn hóa X theo dòng
        X = X / X.sum(axis=1, keepdims=True)

        for t in range(self.iterations_num - 1):
            G = np.empty_like(X)

            for i in range(K):
                # Công thức đạo hàm
                A = (X[:, i] @ (self.W @ np.ones(N))) * (self.W @ np.ones(N) - 2 * (self.W @ X[:, i]))
                B = (X[:, i] @ (self.W @ (np.ones(N) - X[:, i]))) * (self.W @ np.ones(N))
                C = (X[:, i] @ (self.W @ np.ones(N))) ** 2
                g = (A - B) / C

                G[:, i] = g

            Y = X * np.exp(-self.step_size * G)
            X = Y / Y.sum(axis = 1, keepdims = True)

        return np.argmax(X, axis=1) + 1

class PartitioningUnbalancedGraph(): # Algorithm 3
    def __init__(self, partition_scheme: PartitionScheme, site_num: int):
        self.partition_scheme = partition_scheme
        self.site_num = site_num

    def run(self) -> PartitionScheme:
        return self.partition_scheme

class PartitioningMinimumCutGraph(): # Algorithm 4
    def __init__(self, partition_scheme: PartitionScheme, site_num: int, depth_search: int = 2):
        self.partition_scheme = partition_scheme
        self.site_num = site_num
        self.depth_search = depth_search

    def run(self) -> PartitionScheme:
        return self.partition_scheme

class FirstPartitioning(): # Algorithm 5
    def __init__(self, partition_scheme: PartitionScheme, site_num: int):
        self.partition_scheme = partition_scheme
        self.site_num = site_num

    def run(self) -> PartitionScheme:
        return self.partition_scheme

class Partitioning(): # Algorithm 2
    def __init__(self, partition_scheme: PartitionScheme, site_num: int, workload_factor: float, distributed_percentage: float = 0.02):
        self.partition_scheme = partition_scheme
        self.site_num = site_num
        self.workload_factor = workload_factor
        self.distributed_percentage = distributed_percentage

    def getLoads(self, partition_scheme: PartitionScheme):
        loads = [
            sum(partition_scheme.heat_graph.nodes[v]["weight"] for v in vertices)
            for vertices in partition_scheme.partitions.values()
        ]
        return loads

    def getCriticalEdgesLoad(self, partition_scheme: PartitionScheme):
        vertex_partitions = {
            v: pid 
            for pid, vertices in partition_scheme.partitions.items()
            for v in vertices
        }
        total = 0
        for u, v, data in partition_scheme.heat_graph.edges(data=True):
            if vertex_partitions[u] != vertex_partitions[v]:
                total += data.get("weight", 0)

        return total

    def run(self) -> PartitionScheme:
        current_scheme = self.partition_scheme
        G = copy.deepcopy(current_scheme)

        loads = self.getLoads(G)
        minPartition = min(loads)
        maxPartition = max(loads)
        sumLoads = sum(loads)
        avgWorkload = sumLoads / len(self.partition_scheme.partitions)

        if (maxPartition > (1 + self.workload_factor) * avgWorkload) or (minPartition < (1 - self.workload_factor) * avgWorkload):
            G = PartitioningUnbalancedGraph(G, self.site_num).run()
        
        criticalEdgesLoad = self.getCriticalEdgesLoad(G)
        if criticalEdgesLoad > self.distributed_percentage * sumLoads:
            G = PartitioningMinimumCutGraph(G, self.site_num).run()

        if len(G.partitions) == 1:
            G = FirstPartitioning(G, self.site_num).run()

        if G.partitions != self.partition_scheme.partitions or len(G.partitions) != self.site_num:
            G = Partitioning(G, self.site_num, self.workload_factor, self.distributed_percentage).run()

        return G




