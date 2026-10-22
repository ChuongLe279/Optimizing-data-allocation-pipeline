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
    def __init__(self, partition_scheme: PartitionScheme, site_num: int, workload_factor: float):
        self.partition_scheme = partition_scheme
        self.site_num = site_num
        self.workload_factor = workload_factor

    def getLoads(self, partition_scheme: PartitionScheme) -> dict[int, float]:
        return {
            pid: sum(
                partition_scheme.heat_graph.nodes[v]["weight"]
                for v in vertices
            )
            for pid, vertices in partition_scheme.partitions.items()
        }

    def mostProeminentTuple(self, partition_scheme: PartitionScheme, overloadedPartition: int):
        """
        Tiêu chí vertex:
            - weight cao nhất
            - weighted degree thấp nhất

        Chọn ra các partition có workload lớn nhất nhưng số kết nối
        cạnh của nó nhỏ nhất. Đỡ tốn chi phí di chuyển.
        """

        graph = partition_scheme.heat_graph
        vertices = partition_scheme.partitions[overloadedPartition]

        candidate = None
        maxWeight = -1
        minDegree = float("inf")

        for v in vertices:
            weight = graph.nodes[v]["weight"]
            degree = graph.degree(v, weight="weight")

            if weight > maxWeight or (weight == maxWeight and degree < minDegree):
                candidate = v
                maxWeight = weight
                minDegree = degree

        # key
        return candidate

    def relocateTuple(self, partition_scheme: PartitionScheme, proeminent_tuple: int, underloadPartition: int, overloadPartition: int):
        """
        Thực hiện các bước sau:
            Chuyển proeminent_tuple sang underloadPartition.
            Di chuyển hàng xóm lân cận đi theo (di chuyển cả nhóm).
            Giới hạn độ tìm kiếm = 2 bước nhảy.

        """
        loads = self.getLoads(partition_scheme)
        avgWorkload = sum(loads.values()) / len(partition_scheme.partitions)

        maxLoad = (1 + self.workload_factor) * avgWorkload
        vertexWeight = partition_scheme.heat_graph.nodes[proeminent_tuple]["weight"]

        """
        Kiểm tra workload của partition đích sau khi nhận vertex.
        Nếu vượt quá maxload thì sẽ không di chuyển
        """
        if loads[underloadPartition] + vertexWeight > maxLoad:
            return False

        # Chuyển proeminent_tuple sang underloadPartition.
        partition_scheme.partitions[overloadPartition].remove(proeminent_tuple)
        partition_scheme.partitions[underloadPartition].add(proeminent_tuple)

        """
        TODO: Áp dụng hai bước dưới đây sẽ tối ưu thuật toán hơn. Nhưng vì
        bài báo không nhắc đến phương pháp cụ thể, áp dụng vào thì sẽ gây ra phức tạp không 
        cần thiết.
        """
        # Di chuyển hàng xóm lân cận đi theo (di chuyển cả nhóm).
        # Giới hạn độ tìm kiếm = 2 bước nhảy.
        return True

    def run(self) -> PartitionScheme:
        current_scheme = self.partition_scheme
        G = copy.deepcopy(current_scheme)

        loads = self.getLoads(G)
        # Partition có tổng workload thấp nhất
        underloadedPartition = min(loads, key=loads.__getitem__)
        # Partition có tổng workload cao nhất
        overloadedPartition = max(loads, key=loads.__getitem__)
        avgWorkload = sum(loads.values()) / len(G.partitions)

        n = ()
        while loads[overloadedPartition] > (1 + self.workload_factor) * avgWorkload:
            """
            Chọn một vertex phù hợp để di chuyển từ partition có workload cao 
            nhất sang thấp nhất.
            """
            n = self.mostProeminentTuple(G, overloadedPartition)
            if n is None:
                break

            relocate_bool = self.relocateTuple(G, n, underloadedPartition, overloadedPartition)
            if not relocate_bool:
                break

            loads = self.getLoads(G)
            underloadedPartition = min(loads, key=loads.__getitem__)
            overloadedPartition = max(loads, key=loads.__getitem__)
            avgWorkload = sum(loads.values()) / len(G.partitions)

        loads = self.getLoads(G)
        # Key
        underloadedPartition = min(loads, key=loads.__getitem__)
        overloadedPartition = max(loads, key=loads.__getitem__)

        if (loads[overloadedPartition] > (1 + self.workload_factor) * avgWorkload or loads[underloadedPartition] < (1-self.workload_factor) * avgWorkload):
            G = PartitioningUnbalancedGraph(G, self.site_num, self.workload_factor).run()

        return G

class PartitioningMinimumCutGraph(): # Algorithm 4
    """
    Giảm giao dịch phân tán.
    """
    def __init__(self, partition_scheme: PartitionScheme, site_num: int, depth_search: int = 2):
        self.partition_scheme = partition_scheme
        self.site_num = site_num
        self.depth_search = depth_search

    def criticalEdges(self, partition_scheme: PartitionScheme) -> list[tuple[int, int, float]]:
        """Trả về danh sách các cạnh nối giữa hai partition khác nhau.
        return (u, v, weight)
        VD: critical_edges = [
                (1, 5, 10),
                (2, 6, 7),
                (3, 8, 4)
            ]
        """
        vertex_partitions = {
            v: pid 
            for pid, vertices in partition_scheme.partitions.items()
            for v in vertices
        }
        critical_edges = []

        for u, v, data in partition_scheme.heat_graph.edges(data=True):
            if vertex_partitions[u] != vertex_partitions[v]:
                critical_edges.append((u, v, float(data["weight"])))

        return critical_edges

    def hottestEdge(self, criticalEdges: list[tuple[int, int, float]]) -> tuple[int, int, float]:
        """Chỉ trả về một cạnh có weight lớn nhất duy nhất."""
        return max(criticalEdges, key=lambda edge: edge[2]) # (u, v, w) -> trả cạnh có w lớn nhất

    def explorePossibility(self, partition_scheme: PartitionScheme, vertex: int, target_partition: int) -> float:
        """
        Đánh giá di chuyển.
        vertex: ID vertex cần chuyển.
        target_partition: ID partition đích.
        """
        raise NotImplementedError()

    def explorePossibilititesReplication(self) -> PartitionScheme:
        raise NotImplementedError()

    def migrateTuple(self, partition_scheme: PartitionScheme, vertex: int, origin_partition: int,target_partition: int) -> PartitionScheme:
        partition_scheme.partitions[origin_partition].remove(vertex)
        partition_scheme.partitions[target_partition].add(vertex)
             
        return partition_scheme

    def run(self) -> PartitionScheme:
        criticalEdges = self.criticalEdges(self.partition_scheme)
        for ce in criticalEdges:
            ce = self.hottestEdge(criticalEdges)
            u, v, weight = ce

            origin_partition = next(
                pid for pid, vertices in self.partition_scheme.partitions.items()
                if u in vertices
            )
            target_partition = next(
                pid for pid, vertices in self.partition_scheme.partitions.items()
                if v in vertices
            )

            chargeTestOrigin = self.explorePossibility(self.partition_scheme, u, target_partition)
            chargeTestTarget =  self.explorePossibility(self.partition_scheme, v, origin_partition)

            if chargeTestOrigin < 0 and chargeTestTarget < 0:
                G = self.explorePossibilititesReplication()

            if chargeTestOrigin > chargeTestTarget and chargeTestTarget >= 0:
                G = self.migrateTuple(self.partition_scheme, v, origin_partition, target_partition)   
            else:
                if chargeTestOrigin >= 0:
                    G = self.migrateTuple(self.partition_scheme, u, origin_partition, target_partition)
                else:
                    G = self.partition_scheme

        return G # type: ignore

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
        avgWorkload = sum(loads) / len(self.partition_scheme.partitions)

        if (maxPartition > (1 + self.workload_factor) * avgWorkload) or (minPartition < (1 - self.workload_factor) * avgWorkload):
            G = PartitioningUnbalancedGraph(G, self.site_num, self.workload_factor).run()

        """
        Tổng trọng số weight của tất cả các cạnh nối giữa 2
        partition khác nhau trong Heat Graph.
        """
        criticalEdgesLoad = self.getCriticalEdgesLoad(G)
        if criticalEdgesLoad > self.distributed_percentage * sum(loads):
            G = PartitioningMinimumCutGraph(G, self.site_num).run()

        if len(G.partitions) == 1:
            G = FirstPartitioning(G, self.site_num).run()

        if G.partitions != self.partition_scheme.partitions or len(G.partitions) != self.site_num:
            G = Partitioning(G, self.site_num, self.workload_factor, self.distributed_percentage).run()

        return G




