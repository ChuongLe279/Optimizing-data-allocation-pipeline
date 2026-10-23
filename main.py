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
        cần thiết. **Nchung là t ko biết làm.
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
    def __init__(self, partition_scheme: PartitionScheme, workload_factor: float, depth_search: int = 2):
        self.partition_scheme = partition_scheme
        self.workload_factor = workload_factor
        self.depth_search = depth_search

    def criticalEdges(self, partition_scheme: PartitionScheme) -> list[tuple[int, int, float]]:
        """
        Trả về danh sách các cạnh nối giữa hai partition khác nhau.
        return (u, v, weight)
        VD: critical_edges = [
                (1, 5, 10),
                (2, 6, 7),
                (3, 8, 4)
            ]
        """
        vertex_partitions = {}

        for pid, vertices in partition_scheme.partitions.items():
            for vertex in vertices:
                if vertex not in vertex_partitions:
                    vertex_partitions[vertex] = set()

                vertex_partitions[vertex].add(pid)

        critical_edges = []

        for u, v, data in partition_scheme.heat_graph.edges(data=True):
            common_partitions = vertex_partitions[u] & vertex_partitions[v]

            if not common_partitions:
                critical_edges.append((u, v, float(data["weight"])))

        return critical_edges

    def hottestEdge(self, criticalEdges: list[tuple[int, int, float]]) -> tuple[int, int, float]:
        """Chỉ trả về một cạnh có weight lớn nhất duy nhất."""
        return max(criticalEdges, key=lambda edge: edge[2]) # (u, v, w) -> trả cạnh có w lớn nhất
    
    def migrateTuple(self, partition_scheme: PartitionScheme, vertices: list[int], origin_partition: int, target_partition: int) -> PartitionScheme:
        for vertex in vertices:
            partition_scheme.partitions[origin_partition].remove(vertex)
            partition_scheme.partitions[target_partition].add(vertex)
        return partition_scheme

    def checkBalance(self, partition_scheme: PartitionScheme) -> bool:
        """
        Kiểm tra Equation 1:
        load(partition) <= (1 + alpha) * average_load
        với mọi partition.
        """
        graph = partition_scheme.heat_graph
        partitions = partition_scheme.partitions
        k = len(partitions)

        if k == 0:
            return False

        # Tổng tải của tất cả đỉnh trong đồ thị
        total_load = sum(
            graph.nodes[v]["weight"]
            for v in graph.nodes
        )

        average_load = total_load / k
        upper_bound = (1 + self.workload_factor) * average_load

        for vertices in partitions.values():
            partition_load = sum(
                graph.nodes[v]["weight"]
                for v in vertices
            )

            if partition_load > upper_bound:
                return False

        return True

    def calculateMinimumCut(self, partition_scheme: PartitionScheme) -> float:
        """Trả về tổng trọng số các cạnh nối giữa các partition."""
        critical_edges = self.criticalEdges(partition_scheme)
        total_weight = 0.0

        for u, v, weight in critical_edges:
            total_weight += weight

        return total_weight

    def explorePossibility(self, partition_scheme: PartitionScheme, vertex: int, origin_partition: int, target_partition: int) -> float:
        """
        Đầu vào:
            vertex: ID đỉnh cần chuyển.
            target_partition: ID partition đích.

        Đầu ra:
            Cạnh nhỏ nhất sau khi di chuyển vertex.

        Các bước làm:
            Di chuyển thử vertex vào target partition (tối đa 2 hops).
            Kiểm tra có vi phạm phương trình 1 (Kiểm soát cân bằng tải giữa partition).
            Tính cạnh nhỏ nhất.
            Lặp lại đến khi tìm được cạnh nhỏ nhất.
        """

        minimal_cut = float('inf') # Cạnh nhỏ nhất sau khi di chuyển vertex.
        heat_graph = partition_scheme.heat_graph
        
        # Các đỉnh cách vertex đúng 1 cạnh
        one_hop = set(heat_graph.neighbors(vertex)) - {vertex}
        # Các đỉnh cách vertex đúng 2 cạnh
        two_hops = set()
        for neighbor in one_hop:
            for next_neighbor in heat_graph.neighbors(neighbor):
                if next_neighbor != vertex and next_neighbor not in one_hop:
                    two_hops.add((vertex, neighbor, next_neighbor))

        trial_scheme = copy.deepcopy(partition_scheme)

        # Chuyển các đỉnh trong phương án sang target_partition
        for partition_pair in two_hops:
            vertices = list(partition_pair)
            trial_scheme = self.migrateTuple(partition_scheme, vertices, origin_partition, target_partition)

            # Kiểm tra cân bằng tải của trial_scheme
            if not self.checkBalance(trial_scheme):
                break

            # Nếu hợp lệ, tính tổng trọng số cạnh cắt rồi so sánh với kết quả tốt nhất
            cut = self.calculateMinimumCut(trial_scheme)
            minimal_cut = min(cut, minimal_cut)

        # Nếu ko có trường hợp nào hợp lệ thì trả về -1
        return min(minimal_cut, -1)

    
    def checkMinPartition(self, original_scheme: PartitionScheme, new_scheme: PartitionScheme) -> bool:
        """Kiểm tra phương án mới có giảm tổng trọng số cạnh cắt không. (đây là equation 2)"""
        original_cut = self.calculateMinimumCut(original_scheme)
        new_cut = self.calculateMinimumCut(new_scheme)
        return new_cut < original_cut

    def createReplication(self, partition_scheme: PartitionScheme, vertex: int, target_partition: int):
        """Thêm vertex vào partition đích, giữ nguyên bản gốc."""
        partition_scheme.partitions[target_partition].add(vertex)
        return partition_scheme

    def explorePossibilititesReplication(self, partition_scheme: PartitionScheme, vertex_one: int, vertex_two: int, partition_one: int, partition_two: int) -> PartitionScheme:
        """
        Tạo bản sao dữ liệu nếu hàm explorePossibility không hợp lệ (trả về -1).

        TODO: Tạo bản sao cả node hàng xóm. Quá khó nên h ko làm đâu.

        Đầu vào:
            vertex_one thuộc partition_one.
            vertex_two thuộc partition_two.

        Bước làm:
            Tạo bảng sao vertex_one ở partition_two.
                Kiểm tra thỏa phương trình 1 và 2.
                Nếu thỏa, trả về partition_scheme và dừng.
            Tạo bảng sao vertex_one ở partition_two.
                Kiểm tra thỏa phương trình 1 và 2.
                Nếu thỏa, trả về partition_scheme và dừng.

        Note:
            Hàm kiểm tra equation 1: checkBalance
            Hàm kiểm tra equation 2: checkMinPartition
        """
        candidates = [
            (vertex_one, partition_two),
            (vertex_two, partition_one)
        ]

        for vertex, target_partition in candidates:
            trial_scheme = copy.deepcopy(partition_scheme)

            self.createReplication(
                trial_scheme,
                vertex,
                target_partition
            )

            if not self.checkBalance(trial_scheme):
                continue

            if not self.checkMinPartition(partition_scheme, trial_scheme):
                continue

            return trial_scheme

        # Không tìm được phương án hợp lệ
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

            chargeTestOrigin = self.explorePossibility(self.partition_scheme, u, origin_partition, target_partition)
            chargeTestTarget =  self.explorePossibility(self.partition_scheme, v, origin_partition, origin_partition)

            if chargeTestOrigin < 0 and chargeTestTarget < 0:
                G = self.explorePossibilititesReplication(self.partition_scheme, u, v, origin_partition, target_partition)

            if chargeTestOrigin > chargeTestTarget and chargeTestTarget >= 0:
                G = self.migrateTuple(self.partition_scheme, [v], origin_partition, target_partition)   
            else:
                if chargeTestOrigin >= 0:
                    G = self.migrateTuple(self.partition_scheme, [u], origin_partition, target_partition)
                else:
                    G = self.partition_scheme

        return G # type: ignore

class FirstPartitioning(): # Algorithm 5
    def __init__(self, partition_scheme: PartitionScheme, site_num: int):
        self.partition_scheme = partition_scheme
        self.heat_graph = partition_scheme.heat_graph
        self.site_num = site_num

    # coarsening graph: Gom cụm các đỉnh projection nối nhau bằng cạnh trọng số 0 (thuộc cùng một tuple) thành một đỉnh đại diện với tổng trọng số nhiệt
    # gom cụm các đỉnh thành 1 tập duy nhất, lúc này chỉ cần quản lí theo từng tập, nhiệt của mỗi tập mới bằng tổng nhiệt của các đỉnh cộng lại
    def _coarsening_graph(self) -> tuple[nx.Graph, dict[int, list[int]], dict[int, int]]:
        zero_graph = nx.Graph()
        zero_graph.add_nodes_from(self.heat_graph.nodes(data=True))
        for u, v, d in self.heat_graph.edges(data=True):
            if float(d.get('weight', 0.0)) == 0.0:
                zero_graph.add_edge(u, v)

        coarse_g = nx.Graph()
        node_to_coarse: dict[int, int] = {}
        coarse_to_nodes: dict[int, list[int]] = {}

        for c_idx, comp in enumerate(nx.connected_components(zero_graph)):
            members = list(comp)
            coarse_to_nodes[c_idx] = members
            total_w = sum(float(self.heat_graph.nodes[n].get('weight', 1.0)) for n in members)
            coarse_g.add_node(c_idx, weight=total_w)
            for n in members:
                node_to_coarse[n] = c_idx

        for u, v, d in self.heat_graph.edges(data=True):
            w = float(d.get('weight', 0.0))
            if w > 0.0:
                cu, cv = node_to_coarse[u], node_to_coarse[v]
                if cu != cv:
                    cur_w = float(coarse_g.get_edge_data(cu, cv, {}).get('weight', 0.0))
                    coarse_g.add_edge(cu, cv, weight=cur_w + w)

        return coarse_g, coarse_to_nodes, node_to_coarse

    def run(self) -> PartitionScheme:
        coarse_g, coarse_to_nodes, _ = self._coarsening_graph()

        partitions: dict[int, set[int]] = {p: set() for p in range(self.site_num)}
        partition_loads: dict[int, float] = {p: 0.0 for p in range(self.site_num)}
        visited: set[int] = set()

        sorted_nodes = sorted(
            coarse_g.nodes(),
            key=lambda n: (float(coarse_g.nodes[n].get('weight', 1.0)), -n),
            reverse=True
        )

        # Hottest Tuples Seed: Phân bổ lần lượt các đỉnh nóng nhất vào từng partition ban đầu
        # tìm ra các đỉnh nóng nhất, chia đều cho các site, tránh tình trạng 1 site quá tải
        for p in range(min(self.site_num, len(sorted_nodes))):
            node = sorted_nodes[p]
            partitions[p].add(node)
            partition_loads[p] += float(coarse_g.nodes[node].get('weight', 1.0))
            visited.add(node)

        # Remaining Tuples Allocation: Lần lượt duyệt các đỉnh còn lại theo độ nóng giảm dần, chọn partition tối ưu dựa trên tổng trọng số cạnh kết nối và mức độ cân bằng tải
        # Ưu tiên xếp đỉnh vào partition có lực liên kết mạnh nhất để tránh việc transaction bị phân tán giữa các partition
        # Nếu không có liên kết hoặc độ liên kết bằng nhau, thuật toán sẽ chọn partition đang có tổng tải thấp nhất để đảm bảo cân bằng tải theo Equation 1
        for node in sorted_nodes:
            if node in visited:
                continue
            node_w = float(coarse_g.nodes[node].get('weight', 1.0))
            best_p = 0
            best_score = None
            for p in range(self.site_num):
                aff = sum(
                    float(coarse_g[node][nbr].get('weight', 0.0))
                    for nbr in coarse_g.neighbors(node)
                    if nbr in partitions[p]
                )
                score = (aff, -partition_loads[p])
                if best_score is None or score > best_score:
                    best_score = score
                    best_p = p
            partitions[best_p].add(node)
            partition_loads[best_p] += node_w 
            visited.add(node)

        # UncoarseningGraph: Bung các đỉnh gom cụm về lại các đỉnh ban đầu trong đồ thị nhiệt
        # và gán nhãn partition tương ứng vào self.partition_scheme.partitions
        final_partitions: dict[int, set[int]] = {p: set() for p in range(self.site_num)}
        for p, c_nodes in partitions.items():
            for cn in c_nodes:
                for orig_node in coarse_to_nodes[cn]:
                    final_partitions[p].add(orig_node)
                    self.heat_graph.nodes[orig_node]['partition'] = p

        self.partition_scheme.partitions = final_partitions
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
            G = PartitioningMinimumCutGraph(G, self.workload_factor).run() # algorithm 4

        if len(G.partitions) == 1:
            G = FirstPartitioning(G, self.site_num).run()

        if G.partitions != self.partition_scheme.partitions or len(G.partitions) != self.site_num:
            G = Partitioning(G, self.site_num, self.workload_factor, self.distributed_percentage).run()

        return G




