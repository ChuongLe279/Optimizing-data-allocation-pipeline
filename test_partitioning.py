"""Algorithm 2: ngưỡng, điều phối và tích hợp; không test/chạy FirstPartitioning.
Chạy: python -B test_partitioning.py. Có benchmark ground truth vét cạn trên đồ thị nhỏ.
"""
import sys
import unittest
from copy import deepcopy
from unittest.mock import patch

sys.dont_write_bytecode = True
from main import Partitioning, PartitioningUnbalancedGraph as Unbalanced, PartitioningMinimumCutGraph as MinCut
from test_unbalanced_graph import GraphCase, CompactRunner, scheme, community_fixture


class TestPartitioning(GraphCase):
    def setUp(self):
        """Chặn nhánh FirstPartitioning để suite chỉ kiểm tra ba class được yêu cầu."""
        guard = patch('main.FirstPartitioning', side_effect=AssertionError('FirstPartitioning nằm ngoài phạm vi test'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_loads(self):
        """Tính tải từ đỉnh theo từng partition, gồm partition rỗng và đỉnh có bản sao."""
        s = scheme({1: .5, 2: 2.5, 3: 4}, {40: {1, 2}, 7: {2, 3}, 90: set()}, [(1, 3, 999)])
        self.assertEqual(Partitioning(s, 3, .2).getLoads(s), [3, 6.5, 0])

    def test_critical_edges_load(self):
        """Tổng cạnh phân tán bỏ cạnh nội bộ và cạnh có hai đầu cùng nơi lưu bản sao."""
        s = scheme({i: 1 for i in range(1, 6)}, {10: {1, 2}, 20: {2, 3}, 30: {4, 5}},
                   [(1, 2, 100), (2, 3, 80), (1, 3, 4), (3, 4, 5), (1, 5, 0)])
        self.assertEqual(Partitioning(s, 3, .2).getCriticalEdgesLoad(s), 9)

    def test_zero_recursion_depth(self):
        """Depth=0 trả bản sao ngay cả khi còn quá tải/cạnh cắt, không gọi thuật toán con."""
        s = scheme({1: 9, 2: 1}, {10: {1}, 20: {2}}, [(1, 2, 5)])
        with patch.object(Unbalanced, 'run', side_effect=AssertionError('Đã hết depth')), \
             patch.object(MinCut, 'run', side_effect=AssertionError('Đã hết depth')):
            result = Partitioning(s, 2, .2).run(recursion_depth=0)
        self.assert_unchanged(result, s)
        self.assertIsNot(result, s)
        self.assertIsNot(result.heat_graph, s.heat_graph)

    def test_exact_load_boundaries_do_not_rebalance(self):
        """Tải đúng hai biên alpha không kích hoạt cân bằng; phương án ổn định giữ nguyên."""
        s = scheme({1: 3, 2: 5}, {10: {1}, 20: {2}})
        with patch.object(Unbalanced, 'run', side_effect=AssertionError('Tải đang nằm trong biên')), \
             patch.object(MinCut, 'run', side_effect=AssertionError('Không có cạnh cắt')):
            result = Partitioning(s, 2, .25).run()
        self.assert_unchanged(result, s)

    def test_upper_load_violation_dispatches_rebalance(self):
        """Vượt biên tải trên phải gọi Unbalanced, giữ đúng site_num và workload_factor."""
        s = scheme({1: 14, 2: 8, 3: 8}, {10: {1}, 20: {2}, 30: {3}})
        with patch.object(Unbalanced, 'run', autospec=True, side_effect=lambda model: model.partition_scheme) as rebalance:
            Partitioning(s, 3, .2).run(recursion_depth=1)
        rebalance.assert_called_once()
        child = rebalance.call_args.args[0]
        self.assertEqual((child.site_num, child.workload_factor), (3, .2))
        self.assert_unchanged(child.partition_scheme, s)

    def test_lower_load_violation_dispatches_rebalance(self):
        """Chỉ vi phạm biên tải dưới vẫn phải gọi Unbalanced theo điều kiện OR của Algorithm 2."""
        s = scheme({1: 12, 2: 12, 3: 6}, {10: {1}, 20: {2}, 30: {3}})
        with patch.object(Unbalanced, 'run', autospec=True, side_effect=lambda model: model.partition_scheme) as rebalance:
            Partitioning(s, 3, .2).run(recursion_depth=1)
        rebalance.assert_called_once()

    def test_distributed_threshold_is_strict(self):
        """Chỉ gọi MinCut khi cut > theta*tổng tải; đúng bằng hoặc dưới ngưỡng thì bỏ qua."""
        for weight, calls in [(1.99, 0), (2, 0), (2.01, 1)]:
            s = scheme({1: 5, 2: 5}, {10: {1}, 20: {2}}, [(1, 2, weight)])
            with patch.object(MinCut, 'run', autospec=True, side_effect=lambda model: model.partition_scheme) as minimum_cut:
                result = Partitioning(s, 2, .2, distributed_percentage=.2).run(recursion_depth=1)
            self.assertEqual(minimum_cut.call_count, calls, f'Cut={weight}, ngưỡng=2')
            self.assert_unchanged(result, s)

    def test_zero_distributed_threshold(self):
        """Theta=0 gọi MinCut khi có cut dương, không gọi khi cut bằng 0."""
        for weight, calls in [(0, 0), (.01, 1)]:
            s = scheme({1: 5, 2: 5}, {10: {1}, 20: {2}}, [(1, 2, weight)])
            with patch.object(MinCut, 'run', autospec=True, side_effect=lambda model: model.partition_scheme) as minimum_cut:
                Partitioning(s, 2, .2, distributed_percentage=0).run(recursion_depth=1)
            self.assertEqual(minimum_cut.call_count, calls)

    def test_balance_then_cut_uses_updated_scheme(self):
        """Cân bằng trước, tính lại cạnh cắt trên phương án mới rồi mới gọi MinCut."""
        s = scheme({1: 4, 2: 2, 3: 2, 4: 2}, {10: {1, 2, 3}, 20: {4}}, [(2, 3, 3)])
        before, events = deepcopy(s), []

        def rebalance(model):
            """Giả lập migration làm xuất hiện cạnh cắt vốn là cạnh nội bộ."""
            events.append('balance')
            model.partition_scheme.partitions = {10: {1, 2}, 20: {3, 4}}
            return model.partition_scheme

        def minimum_cut(model):
            """Xác minh MinCut nhận đúng phương án vừa cân bằng."""
            events.append('cut')
            self.assertEqual(model.partition_scheme.partitions, {10: {1, 2}, 20: {3, 4}})
            self.assertEqual(model.workload_factor, .2)
            return model.partition_scheme

        with patch.object(Unbalanced, 'run', autospec=True, side_effect=rebalance), \
             patch.object(MinCut, 'run', autospec=True, side_effect=minimum_cut):
            result = Partitioning(s, 2, .2, distributed_percentage=.1).run(recursion_depth=1)
        self.assertEqual(events, ['balance', 'cut'])
        self.assert_valid(result, before)
        self.assert_unchanged(s, before)

    def test_changed_scheme_is_rechecked_until_stable(self):
        """Có thay đổi phải đánh giá lại; nếu vòng kế tiếp không đổi thì dừng trước khi hết depth."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2}, 20: {3, 4}},
                   [(i, j, 1) for i in range(1, 5) for j in range(i + 1, 5)])
        target = {10: {1, 3}, 20: {2, 4}}

        def minimum_cut(model):
            """Vòng đầu đổi phân hoạch; các vòng sau trả cùng trạng thái."""
            model.partition_scheme.partitions = deepcopy(target)
            return model.partition_scheme

        with patch.object(MinCut, 'run', autospec=True, side_effect=minimum_cut) as minimum:
            result = Partitioning(s, 2, .5, distributed_percentage=0).run(recursion_depth=8)
        self.assertEqual(minimum.call_count, 2)
        self.assertEqual(result.partitions, target)
        self.assertEqual(s.partitions, {10: {1, 2}, 20: {3, 4}})

    def test_recursion_budget_stops_oscillation(self):
        """Hai phương án luân phiên vẫn phải dừng đúng ngân sách, không đệ quy vô hạn."""
        first, second = {10: {1, 2}, 20: {3, 4}}, {10: {1, 3}, 20: {2, 4}}
        s = scheme({i: 1 for i in range(1, 5)}, first,
                   [(i, j, 1) for i in range(1, 5) for j in range(i + 1, 5)])

        def toggle(model):
            """Giả lập thuật toán con dao động để kiểm tra giới hạn điều phối."""
            current = model.partition_scheme
            current.partitions = deepcopy(second if current.partitions == first else first)
            return current

        with patch.object(MinCut, 'run', autospec=True, side_effect=toggle) as minimum:
            result = Partitioning(s, 2, .5, distributed_percentage=0).run(recursion_depth=3)
        self.assertEqual(minimum.call_count, 3)
        self.assertEqual(result.partitions, second)
        self.assertEqual(s.partitions, first)

    def test_real_rebalance_pipeline(self):
        """Tích hợp chạy thật: cân bằng ba partition, giữ dữ liệu và đầu vào, không cần FirstPartitioning."""
        s = scheme({1: 4, 2: 3, 3: 2, 4: 1, 5: 2}, {10: {1, 2, 3, 4}, 20: {5}, 30: set()})
        before = deepcopy(s)
        result = Partitioning(s, 3, .25).run(recursion_depth=4)
        self.assertNotEqual(result.partitions, before.partitions)
        self.assert_balanced(result, .25)
        self.assert_valid(result, before)
        self.assert_unchanged(s, before)

    def test_real_min_cut_pipeline(self):
        """Tích hợp chạy thật: giảm cạnh cắt, giữ Equation 1 và không làm mất đỉnh/sửa input."""
        s = scheme({i: 1 for i in range(1, 5)}, {10: {1, 2, 3}, 20: {4}},
                   [(1, 4, 9), (1, 2, 2), (2, 3, 7)])
        before = deepcopy(s)
        model = Partitioning(s, 2, .5, distributed_percentage=0)
        result = model.run(recursion_depth=4)
        self.assertLess(model.getCriticalEdgesLoad(result), model.getCriticalEdgesLoad(before))
        self.assert_balanced(result, .5)
        self.assert_valid(result, before, replicas=True)
        self.assert_unchanged(s, before)

    def test_zero_workload_existing_partitions(self):
        """Hai partition đã tồn tại nhưng rỗng: không chia cho 0 hay gọi thuật toán con."""
        s = scheme({}, {10: set(), 20: set()})
        with patch.object(Unbalanced, 'run', side_effect=AssertionError('Không có tải')), \
             patch.object(MinCut, 'run', side_effect=AssertionError('Không có cạnh cắt')):
            result = Partitioning(s, 2, .2).run()
        self.assert_unchanged(result, s)


    def test_ground_truth_balance_and_cut_pipeline(self):
        """Ground truth: từ tải 5/1 chạy cả cân bằng và giảm cut; replica cho phép đạt cut=0."""
        s = community_fixture()
        self.check_ground_truth(Partitioning(s, 2, 1 / 3, distributed_percentage=0), 1 / 3, expected=0)

    def test_ground_truth_feasible_but_hottest_tuple_blocked(self):
        """Ground truth: pipeline phải đối mặt với cách chia khả thi mà lựa chọn đỉnh nóng bị kẹt."""
        s = scheme({1: 5, 2: 3, 3: 2, 4: 2}, {10: {1, 2}, 20: {3, 4}})
        self.check_ground_truth(Partitioning(s, 2, .2, distributed_percentage=0), .2, expected=0)

    def test_ground_truth_strict_balance_pipeline(self):
        """Ground truth: lặp lại pipeline trên tải cân bằng tuyệt đối có đạt nghiệm cut=1 hay vẫn bị kẹt."""
        s = community_fixture(mixed=True)
        self.check_ground_truth(Partitioning(s, 2, 0, distributed_percentage=0), 0, expected=1)


if __name__ == '__main__':
    unittest.main(testRunner=CompactRunner('PARTITIONING'))
