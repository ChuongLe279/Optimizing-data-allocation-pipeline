"""
Algorithm 3, mục 5.4.1 của An Autonomous Hybrid...; có oracle vét cạn đồ thị nhỏ.
Chạy: python -B test_unbalanced_graph.py. Dùng chung tiện ích test với hai file còn lại.
"""
import sys
import unittest
from collections import Counter
from copy import deepcopy
from itertools import product
from unittest.mock import patch

sys.dont_write_bytecode = True
import networkx as nx
from main import PartitionScheme, PartitioningUnbalancedGraph as Unbalanced


def scheme(weights, partitions, edges=()):
    """Tạo heat graph nhỏ, ID tùy ý, tách tải đỉnh khỏi trọng số cạnh."""
    graph = nx.Graph()
    graph.add_nodes_from((v, {'weight': w}) for v, w in weights.items())
    graph.add_weighted_edges_from(edges)
    return PartitionScheme(graph, deepcopy(partitions))


def cut_weight(s):
    """Tính cạnh cắt độc lập với main: hai đầu không có partition lưu chung mới đóng góp."""
    return sum(data['weight'] for u, v, data in s.heat_graph.edges(data=True)
               if not any(u in vertices and v in vertices for vertices in s.partitions.values()))


def exact_scheme(s, alpha, metric='cut', replicas=False):
    """Vét cạn mọi nơi lưu: tối thiểu peak tải hoặc cut thỏa Eq.1; có thể xét replica.

    Mỗi đỉnh có một mask partition khác 0. Không dùng helper của main, không giới hạn
    bởi số hop/trạng thái đầu. Khi hòa điểm, ưu tiên ít bản sao để dễ đọc nghiệm.
    """
    nodes, pids = list(s.heat_graph), list(s.partitions)
    weights = [s.heat_graph.nodes[v]['weight'] for v in nodes]
    k = len(pids)
    limit = (1 + alpha) * sum(weights) / k
    indices = {v: i for i, v in enumerate(nodes)}
    edges = [(indices[u], indices[v], data['weight']) for u, v, data in s.heat_graph.edges(data=True)]
    choices = range(1, 1 << k) if replicas else [1 << p for p in range(k)]
    best, answer = (float('inf'), float('inf')), None
    for masks in product(choices, repeat=len(nodes)):
        loads = [sum(w for w, mask in zip(weights, masks) if mask & (1 << p)) for p in range(k)]
        if metric == 'cut' and max(loads) > limit + 1e-12:
            continue
        score = max(loads) if metric == 'peak' else sum(w for u, v, w in edges if not masks[u] & masks[v])
        key = score, sum(mask.bit_count() for mask in masks) - len(nodes)
        if key < best:
            best = key
            answer = {pid: {v for v, mask in zip(nodes, masks) if mask & (1 << p)} for p, pid in enumerate(pids)}
    return best[0], answer


def community_fixture(mixed=False):
    """Hai tam giác cạnh 10 nối cầu 1; trạng thái đầu trộn cụm hoặc lệch tải."""
    return scheme({i: 1 for i in range(1, 7)},
                  {10: {1, 2, 4}, 20: {3, 5, 6}} if mixed else {10: {1, 2, 3, 4, 5}, 20: {6}},
                  [(1, 2, 10), (1, 3, 10), (2, 3, 10), (4, 5, 10), (4, 6, 10), (5, 6, 10), (3, 4, 1)])


class GraphCase(unittest.TestCase):
    def assert_unchanged(self, actual, before):
        """Kiểm tra partition và toàn bộ thuộc tính graph không bị sửa ngoài ý muốn."""
        self.assertEqual(actual.partitions, before.partitions)
        self.assertTrue(nx.utils.graphs_equal(actual.heat_graph, before.heat_graph))

    def assert_valid(self, result, before, replicas=False):
        """Không mất/thêm đỉnh hoặc partition; migration bảo toàn số bản sao và dữ liệu graph."""
        self.assertIsInstance(result, PartitionScheme)
        self.assertEqual(set(result.partitions), set(before.partitions))
        self.assertTrue(nx.utils.graphs_equal(result.heat_graph, before.heat_graph))
        counts = Counter(v for group in result.partitions.values() for v in group)
        original = Counter(v for group in before.partitions.values() for v in group)
        self.assertEqual(set(counts), set(before.heat_graph.nodes))
        if not replicas:
            self.assertEqual(counts, original)

    def assert_balanced(self, result, alpha):
        """Kiểm tra trực tiếp Equation 1 từ tải đỉnh, không gọi checkBalance của main."""
        limit = (1 + alpha) * sum(nx.get_node_attributes(result.heat_graph, 'weight').values()) / len(result.partitions)
        for pid, vertices in result.partitions.items():
            load = sum(result.heat_graph.nodes[v]['weight'] for v in vertices)
            self.assertLessEqual(load, limit + 1e-12, f'Partition {pid}: tải {load} > giới hạn {limit}')


    def check_ground_truth(self, model, alpha, metric='cut', expected=None):
        """In nghiệm/điểm thật và heuristic; peak chỉ cần đạt ngưỡng, cut so với tối ưu toàn cục."""
        before = deepcopy(model.partition_scheme)
        optimum, partitions = exact_scheme(before, alpha, metric, replicas=metric == 'cut')
        self.assertIsNotNone(partitions, 'Đồ thị test không có nghiệm ground truth khả thi')
        if expected is not None:
            self.assertAlmostEqual(optimum, expected, places=10, msg='Oracle không khớp nghiệm tính tay')
        result = model.run()
        loads = [sum(result.heat_graph.nodes[v]['weight'] for v in group) for group in result.partitions.values()]
        limit = (1 + alpha) * sum(data['weight'] for _, data in before.heat_graph.nodes(data=True)) / len(before.partitions)
        actual = max(loads) if metric == 'peak' else cut_weight(result)
        name = 'tải lớn nhất' if metric == 'peak' else 'cut'
        self.ground_truth_report = [
            f'Ground truth: { {pid: sorted(v) for pid, v in partitions.items()} } | {name} = {optimum:g}', #type: ignore
            f'Thực tế: { {pid: sorted(v) for pid, v in result.partitions.items()} } | {name} = {actual:g}',
            f'Tải = {loads} | giới hạn mỗi partition = {limit:g}',
        ]
        self.assert_valid(result, before, replicas=metric == 'cut')
        self.assert_unchanged(model.partition_scheme, before)
        if metric == 'peak' and optimum > limit + 1e-12:
            self.ground_truth_report.append('Không tồn tại cách chia đạt ngưỡng; không yêu cầu heuristic làm điều bất khả thi.')
            self.assertGreaterEqual(actual + 1e-12, optimum)
            return
        if metric == 'cut':
            self.ground_truth_report.append(
                f'Chênh lệch cut so với tối ưu = {actual - optimum:g} (oracle cho phép replica).'
                if max(loads) <= limit + 1e-12 else 'Phương án quá tải: chưa hợp lệ để so điểm cut với ground truth.')
        self.assert_balanced(result, alpha)
        if metric == 'cut':
            self.assertAlmostEqual(actual, optimum, places=10,
                                   msg='Benchmark ground truth: heuristic chưa đạt tối ưu toàn cục trên ví dụ này')


class TestUnbalancedGraph(GraphCase):
    def test_loads(self):
        """Tải tính từ đỉnh, đúng với ID không liên tiếp, partition rỗng và bản sao."""
        s = scheme({1: .5, 2: 2.5, 3: 4}, {40: {1, 2}, 7: {2, 3}, 90: set()}, [(1, 3, 999)])
        self.assertEqual(Unbalanced(s, 3, .2).getLoads(s), {40: 3, 7: 6.5, 90: 0})

    def test_hottest_then_lowest_weighted_degree(self):
        """Chọn đỉnh nóng nhất; khi hòa tải chọn tổng trọng số cạnh nhỏ nhất, không đếm số cạnh."""
        s = scheme({1: 8, 2: 8, 3: 7, 4: 1, 5: 1}, {10: {1, 2, 3}, 20: {4, 5}},
                   [(1, 4, 9), (2, 4, 1), (2, 5, 1)])
        self.assertEqual(Unbalanced(s, 2, .2).mostProeminentTuple(s, 10), 2)

    def test_empty_source_has_no_candidate(self):
        """Partition rỗng không có đỉnh ứng viên; trả None thay vì phát sinh lỗi."""
        s = scheme({1: 1}, {10: set(), 20: {1}})
        self.assertIsNone(Unbalanced(s, 2, .2).mostProeminentTuple(s, 10))

    def test_relocate_at_capacity(self):
        """Cho phép chuyển đỉnh khi tải đích đúng ngưỡng (1+alpha)*average; bảo toàn dữ liệu."""
        s = scheme({1: 4, 2: 2, 3: 2, 4: 2}, {10: {1, 2}, 20: {3, 4}}, [(1, 3, 5)])
        before = deepcopy(s)
        self.assertTrue(Unbalanced(s, 2, .6).relocateTuple(s, 1, 20, 10))
        self.assertEqual(s.partitions, {10: {2}, 20: {1, 3, 4}})
        self.assert_valid(s, before)
        self.assert_balanced(s, .6)

    def test_relocate_rejects_overload_without_mutation(self):
        """Từ chối chuyển đỉnh làm đích quá tải; không sửa một phần partition hoặc graph."""
        s = scheme({1: 4, 2: 2, 3: 2, 4: 2}, {10: {1, 2}, 20: {3, 4}})
        before = deepcopy(s)
        self.assertFalse(Unbalanced(s, 2, .2).relocateTuple(s, 1, 20, 10))
        self.assert_unchanged(s, before)

    def test_balanced_run_returns_independent_copy(self):
        """Đồ thị đã cân bằng giữ nguyên phân hoạch; kết quả không dùng chung graph/set đầu vào."""
        s = scheme({1: 2, 2: 2, 3: 2, 4: 2}, {10: {1, 2}, 20: {3, 4}}, [(1, 2, 3)])
        before = deepcopy(s)
        result = Unbalanced(s, 2, 0).run()
        self.assert_unchanged(result, before)
        self.assertIsNot(result, s)
        self.assertIsNot(result.heat_graph, s.heat_graph)
        result.partitions[10].clear()
        result.heat_graph.nodes[1]['weight'] = 999
        self.assert_unchanged(s, before)

    def test_rebalance_multiple_partitions(self):
        """Chuyển nhiều lần, cập nhật partition nóng/lạnh sau mỗi lần; cuối cùng đạt Equation 1."""
        s = scheme({1: 4, 2: 3, 3: 2, 4: 1, 5: 2}, {10: {1, 2, 3, 4}, 20: {5}, 30: set()})
        before = deepcopy(s)
        result = Unbalanced(s, 3, .25).run()
        self.assertNotEqual(result.partitions, before.partitions)
        self.assert_balanced(result, .25)
        self.assert_valid(result, before)
        self.assert_unchanged(s, before)

    def test_exact_balance_boundary_does_not_move(self):
        """Tải 6/4, alpha=20% nằm đúng hai biên; không thực hiện migration thừa."""
        s = scheme({1: 6, 2: 4}, {10: {1}, 20: {2}})
        with patch.object(Unbalanced, 'relocateTuple', side_effect=AssertionError('Không cần di chuyển')):
            result = Unbalanced(s, 2, .2).run()
        self.assert_unchanged(result, s)

    def test_zero_depth(self):
        """Hết ngân sách đệ quy thì trả bản sao hiện tại, không cố di chuyển."""
        s = scheme({1: 9, 2: 1}, {10: {1, 2}, 20: set()})
        with patch.object(Unbalanced, 'relocateTuple', side_effect=AssertionError('Đã hết depth')):
            result = Unbalanced(s, 2, .2).run(recursion_depth=0)
        self.assert_unchanged(result, s)
        self.assertIsNot(result, s)

    def test_empty_scheme(self):
        """Graph và danh sách partition rỗng được trả lại an toàn."""
        s = scheme({}, {})
        self.assert_unchanged(Unbalanced(s, 0, .2).run(), s)

    def test_unmovable_candidate_stops(self):
        """Đỉnh nóng quá lớn để chuyển: dừng trong ngân sách, không mất đỉnh hay chuyển vượt tải."""
        s = scheme({1: 9, 2: 1}, {10: {1, 2}, 20: set()})
        before = deepcopy(s)
        result = Unbalanced(s, 2, .2).run(recursion_depth=3)
        self.assert_unchanged(result, before)
        self.assert_unchanged(s, before)

    def test_partition_load_ties_use_smallest_id(self):
        """Theo mục 5.4.1, hòa tải phải chọn ID partition nhỏ nhất, không theo thứ tự dict."""
        s = scheme({1: 4, 2: 4, 3: 4, 4: 4, 5: 2, 6: 2},
                   {30: {1, 2}, 10: {3, 4}, 40: {5}, 20: {6}})
        with patch.object(Unbalanced, 'relocateTuple', autospec=True, side_effect=Unbalanced.relocateTuple) as move:
            Unbalanced(s, 4, .2).run(recursion_depth=2)
        self.assertTrue(move.called)
        self.assertEqual(move.call_args_list[0].args[3:], (20, 10),
                         'Lần đầu phải chuyển từ partition 10 sang 20 khi hòa tải')


    def test_ground_truth_feasible_three_sites(self):
        """Ground truth: 6 đỉnh/3 site có nghiệm peak=5; heuristic chỉ cần đạt ngưỡng tải 6."""
        s = scheme({1: 4, 2: 3, 3: 3, 4: 2, 5: 2, 6: 1}, {10: {1, 2, 3}, 20: {4, 5}, 30: {6}})
        self.check_ground_truth(Unbalanced(s, 3, .2), .2, metric='peak', expected=5)

    def test_ground_truth_smaller_tuple_can_balance(self):
        """Ground truth: đỉnh nóng nhất không chuyển được nhưng chuyển đỉnh nhỏ hơn có nghiệm hợp lệ."""
        s = scheme({1: 5, 2: 3, 3: 2, 4: 2}, {10: {1, 2}, 20: {3, 4}})
        self.check_ground_truth(Unbalanced(s, 2, .2), .2, metric='peak', expected=7)

    def test_ground_truth_infeasible_capacity(self):
        """Ground truth: đỉnh tải 9 vượt ngưỡng 6 ở mọi cách chia; xác nhận bài toán bất khả thi."""
        s = scheme({1: 9, 2: 1}, {10: {1, 2}, 20: set()})
        self.check_ground_truth(Unbalanced(s, 2, .2), .2, metric='peak', expected=9)


class CompactResult(unittest.TestResult):
    """Một dòng/test, PASS xanh và FAIL/ERROR đỏ; không che giấu lỗi bằng xfail."""
    def __init__(self):
        """Theo dõi kết quả từng bộ test để hỗ trợ chạy chung trong full_test.py."""
        super().__init__()
        self.groups = {}
        self.section = None
        self.grouped = False

    def startTest(self, test):
        """Tách tiêu đề khi chuyển sang bộ test khác, vẫn đánh số liên tục."""
        super().startTest(test)
        name = test.__class__.__module__
        if self.grouped and name != self.section:
            print(f'\n--- {name} ---', flush=True)
            self.section = name

    def show(self, test, status, error=None):
        """In mô tả và lý do ngắn, không in traceback dài."""
        color = '\033[92m' if status == 'PASS' else '\033[91m'
        print(f'{self.testsRun:02d}. [{color}{status:5}\033[0m] {test.shortDescription() or test.id()}', flush=True)
        self.groups.setdefault(test.__class__.__module__, Counter())[status] += 1
        for line in getattr(test, 'ground_truth_report', []):
            print(f'            {line}', flush=True)
        if error:
            print(f"            Lý do: {' '.join(str(error[1]).split()) or error[0].__name__}", flush=True)

    def addSuccess(self, test):
        """In PASS chỉ sau khi test hoàn thành thành công."""
        super().addSuccess(test)
        self.show(test, 'PASS')

    def addFailure(self, test, err):
        """Ghi nhận assertion thất bại và in FAIL."""
        super().addFailure(test, err)
        self.show(test, 'FAIL', err)

    def addError(self, test, err):
        """Ghi nhận exception ngoài assertion và in ERROR."""
        super().addError(test, err)
        self.show(test, 'ERROR', err)


class CompactRunner:
    def __init__(self, title, grouped=False):
        """Đặt tiêu đề theo class đang được kiểm tra."""
        self.title = title
        self.grouped = grouped

    def run(self, suite):
        """Chạy toàn bộ suite và giữ exit code thất bại nếu có lỗi."""
        print(f'KIỂM TRA {self.title}\n', flush=True)
        result = CompactResult()
        result.grouped = self.grouped
        suite.run(result)
        failed, errors = len(result.failures), len(result.errors)
        passed = result.testsRun - failed - errors - len(result.skipped)
        if self.grouped:
            print('\nKẾT QUẢ TỪNG BỘ:', flush=True)
            for name, counts in result.groups.items():
                print(f"  {name:24} | \033[92m{counts['PASS']:2} PASS\033[0m | "
                      f"\033[91m{counts['FAIL']:2} FAIL\033[0m | {counts['ERROR']} ERROR", flush=True)
        print(f'\nTỔNG KẾT: {result.testsRun} test | \033[92m{passed} PASS\033[0m | '
              f'\033[91m{failed} FAIL\033[0m | {errors} ERROR', flush=True)
        return result


if __name__ == '__main__':
    unittest.main(testRunner=CompactRunner('PARTITIONING UNBALANCED GRAPH'))
