"""Algorithm 4, Equations 1-2; kiểm tra hành vi và ground truth vét cạn có replication.
Chạy: python -B test_min_cut_graph.py. Tiện ích dùng chung nằm trong test_unbalanced_graph.py.
"""
import sys
from copy import deepcopy
from unittest.mock import patch
import unittest

sys.dont_write_bytecode = True
from main import PartitioningMinimumCutGraph as MinCut
from test_unbalanced_graph import GraphCase, CompactRunner, scheme, community_fixture


class TestMinimumCutGraph(GraphCase):
    def migration_fixture(self):
        """Một đỉnh có thể chuyển để giảm cạnh cắt mà vẫn nằm trong ngưỡng tải."""
        return scheme({1: 1, 2: 1, 3: 1, 4: 1}, {10: {1, 2, 3}, 20: {4}},
                      [(1, 4, 9), (1, 2, 2), (2, 3, 7)])

    def depth_fixture(self):
        """Chuỗi có lựa chọn khác nhau ở 0/1/2 hop và một đỉnh nằm ngoài 2 hop."""
        return scheme({1: 1, 2: 1, 3: 1, 4: 1, 5: 4}, {10: {1, 2, 3, 5}, 20: {4}},
                      [(1, 4, 10), (1, 2, 8), (2, 3, 7), (3, 5, 1)])

    def test_critical_edges_with_replicas(self):
        """Cạnh là local nếu hai đầu có ít nhất một partition chung, kể cả có bản sao."""
        s = scheme({i: 1 for i in range(1, 6)}, {10: {1, 2}, 20: {2, 3}, 30: {4, 5}},
                   [(1, 2, 100), (2, 3, 80), (1, 3, 4), (3, 4, 5)])
        model = MinCut(s, .5)
        edges = {frozenset((u, v)): w for u, v, w in model.criticalEdges(s)}
        self.assertEqual(edges, {frozenset((1, 3)): 4, frozenset((3, 4)): 5})
        self.assertEqual(model.calculateMinimumCut(s), 9)

    def test_zero_weight_is_not_distributed_transaction(self):
        """Theo ví dụ mục 5.5, cạnh critical có weight>0; cạnh projection weight=0 không được chọn."""
        s = scheme({1: 1, 2: 1}, {10: {1}, 20: {2}}, [(1, 2, 0)])
        self.assertEqual(MinCut(s, .5).criticalEdges(s), [])

    def test_hottest_edge(self):
        """Chọn cạnh có trọng số lớn nhất bất kể vị trí trong danh sách."""
        s = scheme({}, {})
        self.assertEqual(MinCut(s, .2).hottestEdge([(1, 2, 2), (3, 4, 9), (5, 6, 3)]), (3, 4, 9))

    def test_migrate_group(self):
        """Migration chuyển cả nhóm, xóa ở nguồn, thêm ở đích và giữ nguyên graph."""
        s = self.migration_fixture()
        before = deepcopy(s)
        result = MinCut(s, .5).migrateTuple(s, [1, 2], 10, 20)
        self.assertEqual(result.partitions, {10: {3}, 20: {1, 2, 4}})
        self.assert_valid(result, before)

    def test_balance_boundary_and_overload(self):
        """Equation 1 dùng tải đỉnh: chấp nhận đúng ngưỡng, từ chối vượt ngưỡng và hỗ trợ alpha=0."""
        for weights, alpha, expected in [({1: 6, 2: 4}, .2, True),
                                         ({1: 6.01, 2: 3.99}, .2, False),
                                         ({1: 5, 2: 5}, 0, True)]:
            s = scheme(weights, {10: {1}, 20: {2}}, [(1, 2, 1000)])
            self.assertEqual(MinCut(s, alpha).checkBalance(s), expected)

    def test_balance_uses_unique_graph_heat_with_replicas(self):
        """Bản sao không được làm phình tổng tải gốc để nới ngưỡng Equation 1."""
        s = scheme({i: 2 for i in range(1, 5)}, {10: {1, 2, 3}, 20: {1, 2, 3, 4}})
        self.assertFalse(MinCut(s, .5).checkBalance(s))  # Tổng gốc=8, giới hạn=6, tải đích=8.

    def test_empty_partitions_are_not_balanced_scheme(self):
        """Không có partition thì checkBalance trả False, không chia cho 0."""
        s = scheme({}, {})
        self.assertFalse(MinCut(s, .2).checkBalance(s))

    def test_update_min_cut_filters_candidates_without_mutation(self):
        """Loại nhóm ngoài nguồn/quá tải, chọn cut nhỏ nhất trong các nhóm cho trước; không sửa input."""
        s = self.migration_fixture()
        before = deepcopy(s)
        cut, vertices = MinCut(s, .5).updateMinCut(s, [[4], [1, 2, 3], [1, 2], [1]], 10, 20)
        self.assertEqual((cut, vertices), (2, [1]))
        self.assert_unchanged(s, before)

    def test_no_feasible_migration(self):
        """Khi mọi migration đều quá tải, helper trả inf/[] và explore trả -1/[]."""
        s = scheme({1: 1, 2: 1}, {10: {1}, 20: {2}}, [(1, 2, 5)])
        model = MinCut(s, 0)
        self.assertEqual(model.updateMinCut(s, [[1]], 10, 20), (float('inf'), []))
        self.assertEqual(model.explorePossibility(s, 1, 10, 20), (-1, []))

    def test_invalid_exploration_source(self):
        """Không xét di chuyển sang chính nguồn hoặc đỉnh không nằm trong nguồn."""
        s = self.migration_fixture()
        model = MinCut(s, .5)
        self.assertEqual(model.explorePossibility(s, 4, 10, 20), (-1, []))
        self.assertEqual(model.explorePossibility(s, 1, 10, 10), (-1, []))

    def test_depth_zero(self):
        """depth_search=0 chỉ xét đỉnh gốc, không di chuyển hàng xóm."""
        s = self.depth_fixture()
        self.assertEqual(MinCut(s, .75, depth_search=0).explorePossibility(s, 1, 10, 20), (8, [1]))

    def test_depth_one(self):
        """depth_search=1 không được chọn đỉnh cách gốc hai cạnh."""
        s = self.depth_fixture()
        cut, vertices = MinCut(s, .75, depth_search=1).explorePossibility(s, 1, 10, 20)
        self.assertEqual((cut, set(vertices)), (7, {1, 2}))

    def test_depth_two_and_read_only_search(self):
        """Mặc định 2 hop xét nhóm ba đỉnh, không lấy đỉnh ngoài nguồn/ngoài 2 hop và không sửa input."""
        s = self.depth_fixture()
        before = deepcopy(s)
        cut, vertices = MinCut(s, .75).explorePossibility(s, 1, 10, 20)
        self.assertEqual((cut, set(vertices)), (1, {1, 2, 3}))
        self.assert_unchanged(s, before)

    def test_non_improving_migration_is_not_valid_candidate(self):
        """Theo mục 5.4.1/5.5, migration không giảm cut không hợp lệ; explore phải trả -1."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}},
                   [(1, 3, 1), (1, 2, 10), (3, 4, 10)])
        self.assertEqual(MinCut(s, .5).explorePossibility(s, 1, 10, 20), (-1, []))

    def test_check_min_partition_requires_strict_improvement(self):
        """Cut bằng nhau không phải cải thiện; chấp nhận giảm và từ chối tăng."""
        s = self.migration_fixture()
        model = MinCut(s, .5)
        improved = deepcopy(s)
        model.migrateTuple(improved, [1], 10, 20)
        self.assertFalse(model.checkMinPartition(s, deepcopy(s)))
        self.assertTrue(model.checkMinPartition(s, improved))
        self.assertFalse(model.checkMinPartition(improved, s))

    def test_create_replica_is_idempotent(self):
        """Replication giữ bản gốc, không sửa graph; thêm lại cùng bản sao không nhân đôi tải."""
        s = self.migration_fixture()
        before = deepcopy(s)
        model = MinCut(s, .5)
        model.createReplication(s, 1, 20)
        once = deepcopy(s)
        model.createReplication(s, 1, 20)
        self.assertIn(1, s.partitions[10])
        self.assertIn(1, s.partitions[20])
        self.assert_valid(s, before, replicas=True)
        self.assert_unchanged(s, once)

    def test_replication_accepts_balanced_cut_reduction(self):
        """Chấp nhận replica giảm cut và không vượt tải; việc thử không sửa phương án gốc."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}}, [(1, 3, 9)])
        before = deepcopy(s)
        model = MinCut(s, .5)
        result = model.explorePossibilititesReplication(s, 1, 3, 10, 20)
        self.assertIn(1, result.partitions[10] & result.partitions[20])
        self.assertEqual(model.calculateMinimumCut(result), 0)
        self.assert_balanced(result, .5)
        self.assert_valid(result, before, replicas=True)
        self.assert_unchanged(s, before)

    def test_replication_tries_reverse_direction(self):
        """Replica chiều đầu quá tải thì thử chiều ngược, không giữ thay đổi của lần thử bị loại."""
        s = scheme({1: 2, 2: 1, 3: 1, 4: 2}, {10: {1, 2}, 20: {3, 4}}, [(1, 3, 9)])
        before = deepcopy(s)
        result = MinCut(s, .5).explorePossibilititesReplication(s, 1, 3, 10, 20)
        self.assertEqual(result.partitions, {10: {1, 2, 3}, 20: {3, 4}})
        self.assert_balanced(result, .5)
        self.assert_unchanged(s, before)

    def test_replication_rejects_overload_and_no_improvement(self):
        """Từ chối replica nếu quá tải hoặc không giảm cut, giữ nguyên phân hoạch."""
        for weight, alpha in [(9, 0), (0, .5)]:
            s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}}, [(1, 3, weight)])
            before = deepcopy(s)
            result = MinCut(s, alpha).explorePossibilititesReplication(s, 1, 3, 10, 20)
            self.assert_unchanged(result, before)
            self.assert_unchanged(s, before)

    def test_run_migrates_and_preserves_input(self):
        """Chạy thật phải giảm cut, giữ cân bằng và bảo toàn đỉnh; không sửa đầu vào."""
        s = self.migration_fixture()
        before = deepcopy(s)
        model = MinCut(s, .5)
        result = model.run()
        self.assertLess(model.calculateMinimumCut(result), model.calculateMinimumCut(before))
        self.assert_balanced(result, .5)
        self.assert_valid(result, before)
        self.assert_unchanged(s, before)

    def test_run_selects_better_reverse_migration(self):
        """Khi hai chiều đều hợp lệ, chọn chiều có cut nhỏ hơn, kể cả chiều từ target về origin."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}},
                   [(1, 3, 9), (1, 2, 7), (3, 4, 2)])
        result = MinCut(s, .5).run()
        self.assertIn(3, result.partitions[10])
        self.assertNotIn(3, result.partitions[20])
        self.assertEqual(MinCut(result, .5).calculateMinimumCut(result), 2)
        self.assert_balanced(result, .5)

    def test_run_processes_hottest_edges_first(self):
        """Duyệt cạnh giảm dần trọng số và xét cả hai chiều trước khi thử replica."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}}, [(1, 3, 2), (2, 4, 8)])
        model = MinCut(s, .5)
        with patch.object(model, 'explorePossibility', return_value=(-1, [])) as explore, \
             patch.object(model, 'explorePossibilititesReplication', side_effect=lambda g, *args: g) as replicate:
            model.run()
        self.assertEqual([call.args[1] for call in explore.call_args_list], [2, 4, 1, 3])
        self.assertEqual(replicate.call_count, 2)

    def test_run_falls_back_to_replication_when_migrations_worsen_cut(self):
        """Hai chiều migration đều tăng cut: phải thử replica hợp lệ giảm cut theo Algorithm 4."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}},
                   [(1, 3, 1), (1, 2, 10), (3, 4, 10)])
        result = MinCut(s, .5).run()
        self.assertEqual(MinCut(result, .5).calculateMinimumCut(result), 0,
                         'Migration không cải thiện đang chặn bước thử replication')
        self.assert_balanced(result, .5)
        self.assert_valid(result, s, replicas=True)

    def test_no_critical_edges_returns_copy(self):
        """Không có cạnh cắt thì trả bản sao không đổi, không phát sinh migration hay replica."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}}, [(1, 2, 4), (3, 4, 5)])
        model = MinCut(s, .5)
        with patch.object(model, 'explorePossibility', side_effect=AssertionError('Không có cạnh cần xét')):
            result = model.run()
        self.assert_unchanged(result, s)
        self.assertIsNot(result.heat_graph, s.heat_graph)
        self.assertIsNot(result.partitions[10], s.partitions[10])


    def test_ground_truth_two_disconnected_pairs(self):
        """Ground truth: hai cặp nằm chéo partition; nghiệm hợp lệ cut=0 bằng migration."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}}, [(1, 3, 10), (2, 4, 8)])
        self.check_ground_truth(MinCut(s, .5), .5, expected=0)

    def test_ground_truth_replication_beats_migration(self):
        """Ground truth có replica: migration làm cut tăng nhưng một bản sao đạt cut=0 và đủ tải."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}},
                   [(1, 3, 1), (1, 2, 10), (3, 4, 10)])
        self.check_ground_truth(MinCut(s, .5), .5, expected=0)

    def test_ground_truth_strict_balance_requires_exchange(self):
        """Ground truth: tải 3/3, alpha=0; phải trao đổi đỉnh đồng thời mới đạt cut tối ưu 1."""
        s = community_fixture(mixed=True)
        self.check_ground_truth(MinCut(s, 0), 0, expected=1)


if __name__ == '__main__':
    unittest.main(testRunner=CompactRunner('PARTITIONING MINIMUM CUT GRAPH'))
