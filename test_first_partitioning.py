"""Algorithm 5: coarsening, seed, allocation, uncoarsening và ground truth đồ thị nhỏ.
Chạy: python -B test_first_partitioning.py hoặc python -B full_test.py.
FirstPartitioning sửa input tại chỗ; alpha trong benchmark là ngưỡng đánh giá vì API chưa nhận alpha.
"""
import sys
import unittest
from collections import Counter
from copy import deepcopy
from itertools import product
from unittest.mock import patch

sys.dont_write_bytecode = True
import networkx as nx
from main import FirstPartitioning, PartitionScheme, Partitioning
from test_unbalanced_graph import GraphCase, CompactRunner, scheme, cut_weight


def exact_first(s, k, alpha):
    """Vét cạn phân hoạch không replica, giữ nguyên các nhóm projection nối bằng cạnh 0.

    Tính tải/cut trực tiếp trên graph gốc, không gọi _coarsening_graph của main.
    Chỉ so cut giữa các phương án đạt ngưỡng; trả inf/None nếu không có nghiệm.
    """
    zero = nx.Graph()
    zero.add_nodes_from(s.heat_graph)
    zero.add_edges_from((u, v) for u, v, d in s.heat_graph.edges(data=True) if d.get('weight', 0) == 0)
    atoms = list(nx.connected_components(zero))
    cap = (1 + alpha) * sum(d.get('weight', 1) for _, d in s.heat_graph.nodes(data=True)) / k
    best, answer = float('inf'), None
    for assignment in product(range(k), repeat=len(atoms)):
        parts = {p: set() for p in range(k)}
        for atom, pid in zip(atoms, assignment):
            parts[pid].update(atom)
        loads = [sum(s.heat_graph.nodes[v].get('weight', 1) for v in group) for group in parts.values()]
        if max(loads) > cap + 1e-12:
            continue
        owner = {v: p for p, group in parts.items() for v in group}
        cut = sum(d.get('weight', 0) for u, v, d in s.heat_graph.edges(data=True) if owner[u] != owner[v])
        if cut < best:
            best, answer = cut, parts
    return best, answer


class TestFirstPartitioning(GraphCase):
    def assert_output(self, result, before, k):
        """Đủ khóa site, mỗi đỉnh đúng một nơi, nhãn đúng và dữ liệu gốc được giữ ngoài nhãn partition."""
        self.assertIsInstance(result, PartitionScheme)
        self.assertEqual(set(result.partitions), set(range(k)))
        self.assertEqual(Counter(v for group in result.partitions.values() for v in group),
                         Counter(iter(before.heat_graph)))
        for pid, group in result.partitions.items():
            for v in group:
                self.assertEqual(result.heat_graph.nodes[v]['partition'], pid)
        for u, v, data in before.heat_graph.edges(data=True):
            if data.get('weight', 0) == 0:
                self.assertEqual(result.heat_graph.nodes[u]['partition'], result.heat_graph.nodes[v]['partition'],
                                 f'Nhóm projection bị tách ở cạnh {u}-{v}')
        clean, original = deepcopy(result.heat_graph), deepcopy(before.heat_graph)
        for graph in (clean, original):
            for _, attrs in graph.nodes(data=True):
                attrs.pop('partition', None)
        self.assertTrue(nx.utils.graphs_equal(clean, original), 'Mất cạnh, trọng số hoặc metadata khi bung graph')

    def run_checked(self, s, k):
        """Chạy thật và kiểm tra bảo toàn dữ liệu với snapshot trước khi class sửa input."""
        before = deepcopy(s)
        result = FirstPartitioning(s, k).run()
        self.assert_output(result, before, k)
        return result

    def test_coarsening_transitive_zero_edges_and_maps(self):
        """Gom cả chuỗi cạnh 0; hai chiều mapping đầy đủ, không trùng/mất đỉnh và cộng đúng tải."""
        s = scheme({10: 2, 30: 3, 90: 4, 120: 5}, {99: {10, 30, 90, 120}},
                   [(10, 30, 0), (30, 90, 0), (90, 120, 7)])
        graph, groups, inverse = FirstPartitioning(s, 2)._coarsening_graph()
        self.assertEqual({frozenset(g) for g in groups.values()}, {frozenset({10, 30, 90}), frozenset({120})})
        self.assertEqual(Counter(v for group in groups.values() for v in group), Counter(iter(s.heat_graph)))
        for cid, group in groups.items():
            self.assertTrue(all(inverse[v] == cid for v in group))
            self.assertEqual(graph.nodes[cid]['weight'], sum(s.heat_graph.nodes[v]['weight'] for v in group))
        self.assertEqual(graph[inverse[90]][inverse[120]]['weight'], 7)

    def test_coarsening_sums_parallel_edges(self):
        """Nhiều cạnh gốc giữa hai nhóm phải cộng thành một cạnh, không ghi đè hoặc đếm đôi."""
        s = scheme({i: 1 for i in range(1, 5)}, {9: {1, 2, 3, 4}},
                   [(1, 2, 0), (3, 4, 0), (1, 3, 1.25), (2, 3, 2.5), (2, 4, 3)])
        graph, _, mapping = FirstPartitioning(s, 2)._coarsening_graph()
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertAlmostEqual(graph[mapping[1]][mapping[3]]['weight'], 6.75)

    def test_coarsening_drops_internal_edges_and_self_loops(self):
        """Cạnh dương bên trong nhóm và self-loop không thành cạnh cắt, không cộng vào tải đỉnh."""
        s = scheme({1: 2, 2: 3, 3: 4}, {9: {1, 2, 3}},
                   [(1, 2, 0), (2, 3, 0), (1, 3, 50), (1, 1, 100)])
        graph, _, _ = FirstPartitioning(s, 2)._coarsening_graph()
        self.assertEqual(graph.number_of_nodes(), 1)
        self.assertEqual(graph.number_of_edges(), 0)
        self.assertEqual(sum(d['weight'] for _, d in graph.nodes(data=True)), 9)

    def test_positive_edges_do_not_coarsen(self):
        """Cạnh dương rất nhỏ vẫn khác cạnh 0; đỉnh cô lập cũng phải có trong graph thu gọn."""
        s = scheme({1: 1, 2: 2, 3: 3}, {9: {1, 2, 3}}, [(1, 2, 1e-12)])
        graph, groups, mapping = FirstPartitioning(s, 2)._coarsening_graph()
        self.assertEqual(len(groups), 3)
        self.assertEqual(set(mapping), {1, 2, 3})
        self.assertEqual(graph[mapping[1]][mapping[2]]['weight'], 1e-12)

    def test_missing_weights_follow_defaults(self):
        """Theo mặc định hiện có: đỉnh thiếu weight có tải 1, cạnh thiếu weight được coi là 0."""
        s = scheme({1: 1, 2: 4, 3: 2}, {9: {1, 2, 3}})
        del s.heat_graph.nodes[1]['weight']
        s.heat_graph.add_edge(1, 2)
        graph, _, mapping = FirstPartitioning(s, 2)._coarsening_graph()
        self.assertEqual(mapping[1], mapping[2])
        self.assertEqual(graph.nodes[mapping[1]]['weight'], 5)
        self.run_checked(s, 2)

    def test_coarsening_is_read_only_and_independent(self):
        """Coarsening không sửa input; chỉnh trọng số graph thu gọn cũng không sửa graph gốc."""
        s = scheme({1: 2, 2: 3}, {9: {1, 2}}, [(1, 2, 0)])
        before = deepcopy(s)
        graph, groups, _ = FirstPartitioning(s, 2)._coarsening_graph()
        graph.nodes[next(iter(graph))]['weight'] = 999
        groups[next(iter(groups))].clear()
        self.assert_unchanged(s, before)

    def test_coarsening_preserves_all_group_consistent_cuts(self):
        """Mọi cách chia 4 nhóm vào 2 site phải có cùng cut trên graph gốc và graph thu gọn."""
        s = scheme({i: 1 for i in range(1, 7)}, {9: set(range(1, 7))},
                   [(1, 2, 0), (3, 4, 0), (1, 3, 2), (2, 4, 3), (2, 5, 4), (4, 6, 5), (5, 6, 6)])
        graph, groups, mapping = FirstPartitioning(s, 2)._coarsening_graph()
        for assignment in product(range(2), repeat=len(groups)):
            labels = dict(zip(groups, assignment))
            original_cut = sum(d['weight'] for u, v, d in s.heat_graph.edges(data=True)
                               if labels[mapping[u]] != labels[mapping[v]])
            coarse_cut = sum(d['weight'] for u, v, d in graph.edges(data=True) if labels[u] != labels[v])
            self.assertEqual(coarse_cut, original_cut)

    def test_hottest_seeds_go_to_different_sites(self):
        """K=N: các seed được phân bổ theo tải giảm dần, mỗi seed một site."""
        s = scheme({1: 2, 2: 7, 3: 5}, {9: {1, 2, 3}})
        self.assertEqual(self.run_checked(s, 3).partitions, {0: {2}, 1: {3}, 2: {1}})

    def test_seeds_use_aggregated_projection_heat(self):
        """Nhóm projection tải 4+4 phải được chọn trước đỉnh riêng tải 7."""
        s = scheme({1: 4, 2: 4, 3: 7, 4: 6}, {9: {1, 2, 3, 4}}, [(1, 2, 0)])
        result = self.run_checked(s, 2)
        self.assertEqual(result.partitions[0], {1, 2})
        self.assertIn(3, result.partitions[1])

    def test_remaining_vertices_use_descending_heat(self):
        """Các đỉnh còn lại được xử lý theo tải giảm dần; khi không có cạnh chọn site nhẹ nhất."""
        s = scheme({4: 7, 3: 8, 1: 10, 2: 9}, {9: {1, 2, 3, 4}})
        self.assertEqual(self.run_checked(s, 2).partitions, {0: {1, 4}, 1: {2, 3}})

    def test_affinity_is_sum_of_weights(self):
        """Liên kết vào site là tổng cạnh 3+3=6, phải thắng một cạnh đơn trọng số 5."""
        s = scheme({1: 10, 2: 9, 3: 4, 4: 3}, {9: {1, 2, 3, 4}},
                   [(1, 3, 10), (4, 1, 3), (4, 3, 3), (4, 2, 5)])
        result = self.run_checked(s, 2)
        self.assertEqual(result.partitions, {0: {1, 3, 4}, 1: {2}})

    def test_equal_affinity_chooses_lighter_site(self):
        """Hai site có cùng tổng liên kết thì đỉnh mới vào site đang ít tải hơn."""
        s = scheme({1: 8, 2: 6, 3: 2}, {9: {1, 2, 3}}, [(3, 1, 4), (3, 2, 4)])
        self.assertIn(3, self.run_checked(s, 2).partitions[1])

    def test_equal_affinity_and_load_are_deterministic(self):
        """Hòa cả liên kết và tải thì chọn site có ID nhỏ nhất, không phân bổ ngẫu nhiên."""
        s = scheme({1: 5, 2: 5, 3: 1}, {9: {1, 2, 3}}, [(3, 1, 4), (3, 2, 4)])
        self.assertIn(3, self.run_checked(s, 2).partitions[0])

    def test_uncoarsening_preserves_ids_attributes_and_labels(self):
        """Bung nhóm trả đúng ID gốc, giữ metadata/cạnh và thay nhãn partition cũ bằng nhãn mới."""
        s = scheme({-7: 2.5, 42: 1.5, 1001: 4}, {777: {-7, 42, 1001}}, [(-7, 42, 0), (42, 1001, 3)])
        s.heat_graph.graph['info'] = {'name': 'fixture'}
        s.heat_graph.nodes[-7].update(partition=999, columns=['name', 'age'])
        s.heat_graph[-7][42]['kind'] = 'projection'
        result = self.run_checked(s, 2)
        self.assertEqual(result.heat_graph.nodes[-7]['partition'], result.heat_graph.nodes[42]['partition'])

    def test_updates_supplied_scheme_in_place(self):
        """API hiện tại cập nhật trực tiếp partition_scheme và trả cùng object với nhãn nhất quán."""
        s = scheme({1: 2, 2: 1}, {99: {1, 2}})
        result = self.run_checked(s, 2)
        self.assertIs(result, s)
        self.assertEqual(s.partitions, {0: {1}, 1: {2}})

    def test_repeated_run_is_stable(self):
        """Chạy lại cùng graph không làm mất đỉnh, nhân đôi dữ liệu hoặc đổi kết quả."""
        s = scheme({1: 4, 2: 3, 3: 2, 4: 1}, {9: {1, 2, 3, 4}}, [(1, 3, 8), (2, 4, 7)])
        self.run_checked(s, 2)
        before = deepcopy(s)
        self.run_checked(s, 2)
        self.assert_unchanged(s, before)

    def test_one_site(self):
        """K=1 giữ mọi đỉnh ở site 0, kể cả graph có nhóm projection và cạnh dương."""
        s = scheme({1: 2, 2: 3, 3: 4}, {9: {1, 2, 3}}, [(1, 2, 0), (2, 3, 8)])
        self.assertEqual(self.run_checked(s, 1).partitions, {0: {1, 2, 3}})

    def test_more_sites_than_vertices(self):
        """K>N tạo đủ khóa site nhưng cho phép site rỗng; không nhân bản đỉnh để lấp chỗ."""
        s = scheme({1: 3, 2: 2}, {9: {1, 2}})
        self.assertEqual(self.run_checked(s, 4).partitions, {0: {1}, 1: {2}, 2: set(), 3: set()})

    def test_all_projections_form_one_atom(self):
        """Toàn graph nối bằng cạnh 0 vẫn là một nhóm khi K>1, không xé projection để đủ cụm."""
        s = scheme({1: 1, 2: 2, 3: 3}, {9: {1, 2, 3}}, [(1, 2, 0), (2, 3, 0)])
        self.assertEqual(self.run_checked(s, 3).partitions, {0: {1, 2, 3}, 1: set(), 2: set()})

    def test_empty_graph(self):
        """Graph rỗng trả K site rỗng và các mapping coarsening rỗng, không phát sinh lỗi."""
        s = scheme({}, {})
        graph, groups, mapping = FirstPartitioning(s, 3)._coarsening_graph()
        self.assertEqual((len(graph), groups, mapping), (0, {}, {}))
        self.assertEqual(self.run_checked(s, 3).partitions, {0: set(), 1: set(), 2: set()})

    def test_zero_heat_vertices(self):
        """Đỉnh tải 0 vẫn được phân bổ đúng một lần và gán nhãn; không yêu cầu đủ K cụm có tải."""
        s = scheme({i: 0 for i in range(5)}, {9: set(range(5))}, [(0, 1, 0)])
        self.run_checked(s, 3)

    def assert_invalid_sites(self, k):
        """Số site không dương phải bị từ chối rõ bằng ValueError trước khi sửa phân hoạch."""
        s = scheme({1: 1}, {9: {1}})
        before = deepcopy(s)
        try:
            FirstPartitioning(s, k).run()
        except Exception as error:
            self.assertIsInstance(error, ValueError, f'K={k}: cần ValueError rõ ràng, nhận {type(error).__name__}')
        else:
            self.fail(f'K={k} không phải số site hợp lệ')
        self.assert_unchanged(s, before)

    def test_zero_sites_rejected(self):
        """K=0 với graph có dữ liệu phải được từ chối thay vì gây KeyError trong allocation."""
        self.assert_invalid_sites(0)

    def test_negative_sites_rejected(self):
        """K âm phải được từ chối trước khi chạy coarsening/allocation."""
        self.assert_invalid_sites(-1)

    def test_fractional_site_count_rejected(self):
        """K không nguyên không được dùng làm số site; input giữ nguyên khi bị từ chối."""
        s = scheme({1: 1}, {9: {1}})
        before = deepcopy(s)
        with self.assertRaises((TypeError, ValueError)):
            FirstPartitioning(s, 1.5).run()
        self.assert_unchanged(s, before)

    def test_partitioning_pipeline_calls_first_on_unpartitioned_graph(self):
        """Tích hợp: Partitioning gọi First đúng một lần từ một partition, bảo vệ input của pipeline."""
        s = scheme({1: 4, 2: 4, 3: 2, 4: 2}, {99: {1, 2, 3, 4}}, [(1, 3, 10), (2, 4, 10)])
        before = deepcopy(s)
        with patch.object(FirstPartitioning, 'run', autospec=True, side_effect=FirstPartitioning.run) as first:
            result = Partitioning(s, 2, 0, distributed_percentage=0).run(recursion_depth=4)
        first.assert_called_once()
        self.assert_output(result, before, 2)
        self.assert_balanced(result, 0)
        self.assertEqual(cut_weight(result), 0)
        self.assert_unchanged(s, before)

    def compare_ground_truth(self, s, k, alpha, expected):
        """Benchmark độc lập: oracle không replica, giữ projection; alpha chỉ là ngưỡng đánh giá."""
        before = deepcopy(s)
        optimum, truth = exact_first(before, k, alpha)
        if expected is None:
            self.assertIsNone(truth, 'Ví dụ này phải bất khả thi khi giữ nguyên nhóm projection')
            self.assertEqual(optimum, float('inf'))
        else:
            self.assertIsNotNone(truth)
            self.assertAlmostEqual(optimum, expected, places=10, msg='Oracle không khớp nghiệm tính tay')
        result = FirstPartitioning(s, k).run()
        loads = [sum(result.heat_graph.nodes[v]['weight'] for v in group) for group in result.partitions.values()]
        limit = (1 + alpha) * sum(d['weight'] for _, d in before.heat_graph.nodes(data=True)) / k
        cut = cut_weight(result)
        self.ground_truth_report = [
            f'Ground truth: {None if truth is None else {p: sorted(v) for p, v in truth.items()}} | cut = {optimum:g}',
            f'Thực tế: { {p: sorted(v) for p, v in result.partitions.items()} } | cut = {cut:g}',
            f'Tải = {loads} | ngưỡng đánh giá = {limit:g} (alpha={alpha:g}; class chưa nhận alpha).',
        ]
        self.assert_output(result, before, k)
        if truth is None:
            self.ground_truth_report.append('Không có nghiệm trong ngưỡng khi giữ nhóm projection; không đánh FAIL vì bất khả thi.')
            return
        if max(loads) > limit + 1e-12:
            self.ground_truth_report.append('Phân hoạch quá tải: cut thấp hơn cũng chưa phải nghiệm hợp lệ của benchmark.')
        else:
            self.ground_truth_report.append(f'Chênh lệch cut so với tối ưu = {cut - optimum:g}.')
        self.assert_balanced(result, alpha)
        self.assertAlmostEqual(cut, optimum, places=10, msg='Benchmark: cách chọn seed/allocation chưa đạt tối ưu toàn cục')

    def test_ground_truth_two_weighted_pairs(self):
        """Ground truth: hai cặp có tổng tải 6/6, heuristic đạt cut=0 với alpha=0."""
        s = scheme({1: 4, 2: 4, 3: 2, 4: 2}, {9: {1, 2, 3, 4}}, [(1, 3, 10), (2, 4, 10)])
        self.compare_ground_truth(s, 2, 0, expected=0)

    def test_ground_truth_projection_groups(self):
        """Ground truth: 8 projection/4 nhóm, giữ nguyên nhóm và đạt cut=0 với tải 4/4."""
        s = scheme({i: 1 for i in range(1, 9)}, {9: set(range(1, 9))},
                   [(1, 2, 0), (3, 4, 0), (5, 6, 0), (7, 8, 0), (1, 5, 9), (3, 7, 8)])
        self.compare_ground_truth(s, 2, 0, expected=0)

    def test_ground_truth_hottest_seeds_should_share_site(self):
        """Ground truth: hai đỉnh nóng nhất cùng một cộng đồng; ép seed khác site làm bỏ lỡ cut=0."""
        s = scheme({1: 4, 2: 3, 3: 2, 4: 1}, {9: {1, 2, 3, 4}}, [(1, 2, 10), (3, 4, 9)])
        self.compare_ground_truth(s, 2, .4, expected=0)

    def test_ground_truth_affinity_can_overload(self):
        """Ground truth: có cách chia tải 9/9, nhưng ưu tiên liên kết tuyệt đối có thể tạo tải 13/5."""
        s = scheme({1: 6, 2: 5, 3: 4, 4: 3}, {9: {1, 2, 3, 4}}, [(1, 3, 10), (1, 4, 9), (2, 3, 1)])
        self.compare_ground_truth(s, 2, 0, expected=10)

    def test_ground_truth_indivisible_projection_is_infeasible(self):
        """Ground truth: nhóm projection tải 6 lớn hơn ngưỡng 4, mọi cách chia giữ nhóm đều bất khả thi."""
        s = scheme({1: 2, 2: 2, 3: 2, 4: 2}, {9: {1, 2, 3, 4}}, [(1, 2, 0), (2, 3, 0)])
        self.compare_ground_truth(s, 2, 0, expected=None)


if __name__ == '__main__':
    unittest.main(testRunner=CompactRunner('FIRST PARTITIONING'))
