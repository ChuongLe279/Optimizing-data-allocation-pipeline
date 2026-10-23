import networkx as nx
import numpy as np
from scipy.sparse import csr_array
import copy
import random
from faker import Faker


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

        for t in range(self.iterations_num):
            G = np.empty_like(X)

            for i in range(K):
                # Công thức đạo hàm
                A = (X[:, i] @ (self.W @ np.ones(N))) * (self.W @ np.ones(N) - 2 * (self.W @ X[:, i]))
                B = (X[:, i] @ (self.W @ (np.ones(N) - X[:, i]))) * (self.W @ np.ones(N))
                C = (X[:, i] @ (self.W @ np.ones(N))) ** 2
                """
                Chưa xét trường hợp C bằng 0 nữa.
                Có thể cộng một số siêu nhỏ như 1e-10.
                Nhưng mà sợ sai công thức đạo hàm.
                """
                g = (A - B) / C

                G[:, i] = g

            """
            Sợ hàm mũ cao quá gây ra inf.
            Nhưng t ko biết chỉnh công thức gốc như nào.
            """
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
        total_heat = sum(
            partition_scheme.heat_graph.nodes[v]["weight"]
            for v in partition_scheme.heat_graph.nodes
        )

        avgWorkload = total_heat / len(partition_scheme.partitions)

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
            Cạnh nhỏ nhất sau khi di chuyển đỉnh.

        Các bước làm:
            Di chuyển thử vertex vào target partition (tối đa 2 hops).
            Kiểm tra có vi phạm phương trình 1 (Kiểm soát cân bằng tải giữa partition).
            Tính cạnh nhỏ nhất.
            Lặp lại đến khi tìm được cạnh nhỏ nhất.

        Note:
            Chúng ta phải xét cả neighbor cách 1 cạnh và 2 cạnh.
                one_hop = set()
                    Tính min
                two_hop = set()
                    Tính min
            
            Code sẽ lặp lại 1 chút cho dễ debug chứ giờ nhiều hàm quá r.
        """

        minimal_cut = float('inf') # Cạnh nhỏ nhất sau khi di chuyển vertex.
        heat_graph = partition_scheme.heat_graph
        source_vertices = partition_scheme.partitions[origin_partition]
        # Kiểm tra xem các đỉnh đều thuộc original partition hay không.
        if vertex not in source_vertices or origin_partition == target_partition:
            return -1
        """
        Tính min cho 1 hop
        """
        # Các đỉnh cách vertex đúng 1 cạnh
        one_hop = set(heat_graph.neighbors(vertex)) - {vertex}
        for neighbor in one_hop:
            vertices = [vertex, neighbor]
            if not set(vertices).issubset(source_vertices):
                continue
            trial_scheme = copy.deepcopy(partition_scheme)
            trial_scheme = self.migrateTuple(trial_scheme, vertices, origin_partition, target_partition)

            # Kiểm tra cân bằng tải của trial_scheme
            if not self.checkBalance(trial_scheme):
                continue

            # Nếu hợp lệ, tính tổng trọng số cạnh cắt rồi so sánh với kết quả tốt nhất
            cut = self.calculateMinimumCut(trial_scheme)
            minimal_cut = min(cut, minimal_cut)

            trial_scheme = copy.deepcopy(partition_scheme)
        """
        Tính min cho 2 hop
        """
        # Các đỉnh cách vertex đúng 2 cạnh
        two_hops = set()
        for neighbor in one_hop:
            for next_neighbor in heat_graph.neighbors(neighbor):
                if next_neighbor != vertex and next_neighbor not in one_hop:
                    two_hops.add((vertex, neighbor, next_neighbor))

        trial_scheme = copy.deepcopy(partition_scheme)
        # Chuyển các đỉnh trong phương án sang target_partition
        for partition_group in two_hops:
            vertices = list(partition_group) # List chứa đỉnh ID.
            if not set(vertices).issubset(source_vertices):
                continue
            trial_scheme = copy.deepcopy(partition_scheme)
            trial_scheme = self.migrateTuple(trial_scheme, vertices, origin_partition, target_partition)

            # Kiểm tra cân bằng tải của trial_scheme
            if not self.checkBalance(trial_scheme):
                continue

            # Nếu hợp lệ, tính tổng trọng số cạnh cắt rồi so sánh với kết quả tốt nhất
            cut = self.calculateMinimumCut(trial_scheme)
            minimal_cut = min(cut, minimal_cut)

        # Nếu ko có trường hợp nào hợp lệ thì trả về -1
        return -1 if minimal_cut == float("inf") else minimal_cut

    
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
        # Xếp trọng số từ nhỏ đến lớn.
        G = copy.deepcopy(self.partition_scheme)
        criticalEdges = sorted(
            self.criticalEdges(self.partition_scheme),
            key=lambda edge: edge[2],
            reverse=True
        )
        for ce in criticalEdges:
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
            chargeTestTarget =  self.explorePossibility(self.partition_scheme, v, target_partition, origin_partition)

            if chargeTestOrigin < 0 and chargeTestTarget < 0:
                G = self.explorePossibilititesReplication(self.partition_scheme, u, v, origin_partition, target_partition)
                continue

            if chargeTestOrigin > chargeTestTarget and chargeTestTarget >= 0:
                G = self.migrateTuple(self.partition_scheme, [v], target_partition, origin_partition)   
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
        calculator = PartitioningMinimumCutGraph(
            partition_scheme,
            self.workload_factor
        )
        return calculator.calculateMinimumCut(partition_scheme)

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


class BPG_PartitionScheme():
    def __init__(self, heat_graph: nx.Graph, labels: np.ndarray, coarse_to_nodes: dict[int, list[int]] | None = None):
        self.heat_graph = heat_graph
        self.labels = labels
        self.coarse_to_nodes = coarse_to_nodes

    # chuyển nhãn phân cụm của bpg ngược lại thành PartitionScheme
    def run(self) -> PartitionScheme:
        from main import PartitionScheme
        partitions: dict[int, set[int]] = {}
        for idx, cluster_id in enumerate(self.labels):
            p = int(cluster_id)
            if p not in partitions:
                partitions[p] = set()
            if self.coarse_to_nodes:
                orig_nodes = self.coarse_to_nodes[idx]
                partitions[p].update(orig_nodes)
            else:
                partitions[p].add(idx + 1)
        return PartitionScheme(self.heat_graph, partitions)


class CreatePartitionScheme():
    """dữ liệu giả, sau này sẽ chuyển thành hàm Database -> Partitioning Scheme"""
    def __init__(self, num_records: int = 10, seed: int = 42):
        self.num_records = num_records
        self.seed = seed
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        self.data: list[dict] = []
        self.transactions: list[dict] = []
        self.heat_graph: nx.Graph | None = None

    # mock database, thông tin các khách hàng, mỗi người có mã số ID, tên, email, số điện thoại, số dư tiền
    def generate_mock_data(self) -> list[dict]:
        self.data = []
        for i in range(1, self.num_records + 1):
            record = {
                'id': i,
                'name': self.fake.name(),
                'email': self.fake.email(),
                'phone': self.fake.phone_number(),
                'city': self.fake.city(),
                'balance': round(float(self.fake.random_number(digits=5)), 2)
            }
            self.data.append(record)
        return self.data

    # hàm này tạo nhật ký người dùng truy cập, đại khái là vậy
    # ví dụ thao tác người dùng bấm trên web/app
    #
    # ý nghĩa của bảng này là nó ghi lại xem những ai hay được truy cập cùng nhau trong các giao dịch.
    # Thuật toán của chúng ta sẽ nhìn vào bảng này để gom những người hay giao dịch chung về cùng một máy chủ,
    # nhằm triệt tiêu tối đa các giao dịch phân tán làm chậm hệ thống
    def generate_mock_transactions(self, num_transactions: int = 10) -> list[dict]:
        if not self.data:
            self.generate_mock_data()

        self.transactions = []
        half = max(1, self.num_records // 2)
        group_1 = [r['id'] for r in self.data if r['id'] <= half]
        group_2 = [r['id'] for r in self.data if r['id'] > half]

        tx_id = 1
        # cluster 1 transactions
        for _ in range(num_transactions // 2):
            ops = []
            attrs = ['name', 'email'] if random.random() < 0.6 else ['balance']
            picked_ids = random.sample(group_1, min(2, len(group_1)))
            for pid in picked_ids:
                ops.append({'tuple_id': pid, 'attrs': attrs})
            self.transactions.append({'tx_id': tx_id, 'ops': ops})
            tx_id += 1

        # cluster 2 transactions
        for _ in range(max(1, num_transactions // 2 - 1)):
            ops = []
            attrs = ['name', 'email'] if random.random() < 0.6 else ['balance']
            picked_ids = random.sample(group_2, min(2, len(group_2)))
            for pid in picked_ids:
                ops.append({'tuple_id': pid, 'attrs': attrs})
            self.transactions.append({'tx_id': tx_id, 'ops': ops})
            tx_id += 1

        # giả sử có 5 người, chia ra 2 nhóm
        # đoạn code dưới lấy ngẫu nhiên 1 người nhóm 1 và 1 người nhóm 2 nhét chung vào 1 giao dịch, tạo ra giao dịch bắc cầu
        # 2 nhóm này đại diện cho 2 node trong hệ thống phân tán
        # nếu như giao dịch chỉ xảy ra trong 1 nhóm mà (tức là trong 1 node) thì bài toán có gì để xử lý nữa đâu
        if group_1 and group_2:
            self.transactions.append({
                'tx_id': tx_id,
                'ops': [
                    {'tuple_id': random.choice(group_1), 'attrs': ['name', 'email']},
                    {'tuple_id': random.choice(group_2), 'attrs': ['name', 'email']}
                ]
            })

        return self.transactions
    """
    Số đỉnh trên đồ thị: Quyết định bởi số tuple và số loại cột được đọc (phân mảnh dọc)
    Số Node / Site phân tán: Là số máy chủ mà ta muốn chia
    Dù hệ thống có 2 máy chủ hay 10 máy chủ, nếu dữ liệu chỉ bị đọc theo 2 loại cột thì 5 người vẫn chỉ sinh ra đúng 10 đỉnh trên đồ thị nhiệt

    giả sử có 5 người giao dịch với mỗi cụm 2 giao dịch
    đồ thị nhiệt sẽ tạo ra 10 đỉnh
    tương tự nếu có 3 loại phép chiều thì 15
    """
    def build_heat_graph(self) -> nx.Graph:
        if not self.transactions:
            self.generate_mock_transactions()

        # đếm tần suất truy cập cho mỗi tuple projection
        # mỗi tuple projection là một đỉnh trên đồ thị nhiệt
        node_access_counts: dict[tuple[int, tuple[str, ...]], int] = {}
        co_access_counts: dict[tuple[tuple[int, tuple[str, ...]], tuple[int, tuple[str, ...]]], int] = {}

        for tx in self.transactions:
            tx_nodes = []
            for op in tx['ops']:
                t_id = op['tuple_id']
                attrs = tuple(sorted(op['attrs']))
                key = (t_id, attrs)
                node_access_counts[key] = node_access_counts.get(key, 0) + 1
                if key not in tx_nodes:
                    tx_nodes.append(key)

            # đếm tần suất truy cập đồng thời giữa các tuple projection khác nhau trong cùng một giao dịch
            # mỗi cặp tuple projection khác nhau có cùng được truy cập trong 1 giao dịch sẽ tạo thành 1 cạnh
            for i in range(len(tx_nodes)):
                for j in range(i + 1, len(tx_nodes)):
                    pair = tuple(sorted([tx_nodes[i], tx_nodes[j]]))
                    co_access_counts[pair] = co_access_counts.get(pair, 0) + 1

        heat_graph = nx.Graph()
        key_to_id: dict[tuple[int, tuple[str, ...]], int] = {}
        node_id_counter = 1

        # in ra bảng heat graph vertical (đồ thị nhiệt đỉnh)
        # thêm các đỉnh với trọng số nhiệt
        for key in sorted(node_access_counts.keys()):
            key_to_id[key] = node_id_counter
            t_id, attrs = key
            heat_graph.add_node(
                node_id_counter,
                tuple_id=t_id,
                table='customers',
                attrs=list(attrs),
                weight=float(node_access_counts[key])
            )
            node_id_counter += 1

        # in ra bảng heat graph edges (đồ thị nhiệt cạnh)
        # thêm các cạnh với trọng số nhiệt
        for (k1, k2), count in co_access_counts.items():
            u = key_to_id[k1]
            v = key_to_id[k2]
            heat_graph.add_edge(u, v, weight=float(count))

        # thêm cạnh projection dọc cho các tuple giống nhau
        # các cạnh này không có trọng số
        # hai mảnh này là thông tin của cùng 1 người, trọng số bằng 0 để báo cho thuật toán biết 2 phần là của cùng 1 người
        tuple_groups: dict[int, list[int]] = {}
        for key, node_id in key_to_id.items():
            t_id, _ = key
            tuple_groups.setdefault(t_id, []).append(node_id)

        for t_id, group in tuple_groups.items():
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        u = group[i]
                        v = group[j]
                        if not heat_graph.has_edge(u, v):
                            heat_graph.add_edge(u, v, weight=0.0)

        self.heat_graph = heat_graph
        return heat_graph

    def run(self) -> PartitionScheme:
        from main import PartitionScheme
        graph = self.build_heat_graph()
        initial_partitions = {0: set(graph.nodes())}
        return PartitionScheme(graph, initial_partitions)

class PartitionScheme_BPG():
    def __init__(self, partition_scheme, use_coarsening: bool = True):
        self.partition_scheme = partition_scheme
        self.heat_graph = partition_scheme.heat_graph
        self.use_coarsening = use_coarsening
        self.node_to_idx: dict[int, int] = {}
        self.idx_to_node: dict[int, int] = {}
        self.coarse_to_nodes: dict[int, list[int]] = {}
        self.W: csr_array | None = None

    # chuyển đồ thị nhiệt thành ma trận kề thưa w
    def run(self) -> csr_array:
        if self.use_coarsening:
            # gom các mảnh thuộc tính của cùng một tuple lại với nhau
            zero_graph = nx.Graph()
            zero_graph.add_nodes_from(self.heat_graph.nodes(data=True))
            for u, v, d in self.heat_graph.edges(data=True):
                if float(d.get('weight', 0.0)) == 0.0:
                    zero_graph.add_edge(u, v)

            coarse_g = nx.Graph()
            node_to_coarse: dict[int, int] = {}
            for c_idx, comp in enumerate(nx.connected_components(zero_graph)):
                members = list(comp)
                self.coarse_to_nodes[c_idx] = members
                coarse_g.add_node(c_idx)
                for n in members:
                    node_to_coarse[n] = c_idx

            # cộng dồn trọng số giao dịch giữa các tuple sau khi gộp
            for u, v, d in self.heat_graph.edges(data=True):
                w = float(d.get('weight', 0.0))
                if w > 0.0:
                    cu, cv = node_to_coarse[u], node_to_coarse[v]
                    if cu != cv:
                        cur_w = float(coarse_g.get_edge_data(cu, cv, {}).get('weight', 0.0))
                        coarse_g.add_edge(cu, cv, weight=cur_w + w)

            nodes = sorted(list(coarse_g.nodes()))
            self.node_to_idx = {n: i for i, n in enumerate(nodes)}
            self.idx_to_node = {i: n for i, n in enumerate(nodes)}
            n_nodes = len(nodes)
            w_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
            for u, v, d in coarse_g.edges(data=True):
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                w_mat[i, j] = d.get('weight', 1.0)
                w_mat[j, i] = d.get('weight', 1.0)

            self.W = csr_array(w_mat)
            return self.W
        else:
            # chuyển đổi trực tiếp không qua bước gộp tuple
            nodes = sorted(list(self.heat_graph.nodes()))
            self.node_to_idx = {n: i for i, n in enumerate(nodes)}
            self.idx_to_node = {i: n for i, n in enumerate(nodes)}
            n_nodes = len(nodes)
            w_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
            for u, v, d in self.heat_graph.edges(data=True):
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                w = float(d.get('weight', 0.0))
                w_mat[i, j] = w
                w_mat[j, i] = w

            self.W = csr_array(w_mat)
            return self.W