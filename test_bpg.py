"""Algorithm 1, mục 3.1 BPG.pdf. Chạy: python -B test_bpg.py.
Test lỗi số học giữ FAIL, không xfail; không giả định luôn đạt tối ưu toàn cục.
"""
import sys
import unittest
from itertools import product
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
from scipy.sparse import csr_array
from main import BregmanProximalGradient


def objective(W, X):
    """Tính f(X) ở (4a), oracle độc lập với công thức gradient trong main."""
    return sum(x @ W @ (1 - x) / (x @ W.sum(axis=1)) for x in X.T)


def discrete_ncut(W, labels, k):
    """NCut = 1/2 tổng cut/volume của các cụm có đỉnh; bỏ qua nhãn không được dùng."""
    score = 0.0
    for group in range(k):
        inside = labels == group
        if not inside.any():
            continue
        volume = W[inside].sum()
        if volume <= 0:
            return float('inf')
        score += W[np.ix_(inside, ~inside)].sum() / volume
    return score / 2


def exact_partition(W, k):
    """Tìm tối ưu với đúng k cụm để so công bằng với số cụm BPG thực sự tạo ra."""
    best, answer = float('inf'), None
    for tail in product(range(k), repeat=len(W) - 1):
        labels = np.array((0, *tail))
        if len(np.unique(labels)) != k:
            continue
        score = discrete_ncut(W, labels, k)
        if score < best:
            best, answer = score, labels
    return best, answer


class TestBPG(unittest.TestCase):
    def setUp(self):
        """Đồ thị vô hướng có bậc không đều; seed cố định tránh test chập chờn."""
        self.W = csr_array([[0, 9, 1, 0], [9, 0, 2, 1],
                            [1, 2, 0, 7], [0, 1, 7, 0]], dtype=float)
        self.R = np.random.default_rng(42).random((4, 3))

    def run_bpg(self, W=None, R=None, step=.4, iterations=10):
        """Cố định RNG, quan sát X trước argmax; giữ nguyên phép tính thật của class."""
        W = self.W if W is None else W
        R = self.R if R is None else R
        with patch('main.np.random.default_rng') as rng, \
             patch('main.np.argmax', wraps=np.argmax) as rounding, np.errstate(all='ignore'):
            rng.return_value.random.return_value = R.copy()
            labels = BregmanProximalGradient(W, R.shape[1], step, iterations).run()
        rng.return_value.random.assert_called_once_with(R.shape)
        rounding.assert_called_once()
        X = rounding.call_args.args[0].copy()
        self.assertTrue(np.isfinite(X).all(), 'X chứa NaN/Inf trước khi làm tròn')
        self.assertTrue((X >= 0).all(), 'X có xác suất âm')
        np.testing.assert_allclose(X.sum(axis=1), 1, atol=1e-14)
        np.testing.assert_array_equal(labels, X.argmax(axis=1) + 1)
        return labels, X

    def test_output(self):
        """Nhãn là N số nguyên trong [1,K], kể cả K>N; X thuộc simplex."""
        for k in (1, 2, 3, 5):
            labels, _ = self.run_bpg(R=np.random.default_rng(k).random((4, k)))
            self.assertIsInstance(labels, np.ndarray)
            self.assertEqual(labels.shape, (4,))
            self.assertTrue(np.issubdtype(labels.dtype, np.integer))
            self.assertTrue(((labels >= 1) & (labels <= k)).all())

    def test_initialization(self):
        """T=0 chuẩn hóa (1/K + 0.1R) theo hàng và làm tròn thành nhãn bắt đầu từ 1."""
        R = np.array([[0, .9, .1], [.1, .2, .8], [.7, .2, .1], [.3, .3, .3]])
        labels, X = self.run_bpg(R=R, iterations=0)
        expected = 1 / 3 + .1 * R
        np.testing.assert_allclose(X, expected / expected.sum(axis=1, keepdims=True))
        np.testing.assert_array_equal(labels, [2, 3, 1, 1])

    def test_zero_step(self):
        """Bước học 0 giữ nguyên xác suất khởi tạo qua nhiều vòng."""
        _, initial = self.run_bpg(iterations=0)
        _, actual = self.run_bpg(step=0, iterations=20)
        np.testing.assert_allclose(actual, initial, atol=1e-14)

    def test_single_partition(self):
        """K=1: mọi đỉnh có xác suất 1 và nhãn 1."""
        labels, X = self.run_bpg(R=self.R[:, :1], iterations=30)
        np.testing.assert_array_equal(X, np.ones((4, 1)))
        np.testing.assert_array_equal(labels, [1, 1, 1, 1])

    def test_uniform_fixed_point(self):
        """Khởi tạo đều là điểm dừng đối xứng; hòa argmax chọn nhãn đầu tiên."""
        labels, X = self.run_bpg(R=np.zeros((4, 3)), iterations=20)
        np.testing.assert_allclose(X, np.full((4, 3), 1 / 3))
        np.testing.assert_array_equal(labels, [1, 1, 1, 1])

    def test_updates_against_numerical_gradient(self):
        """Đối chiếu 1 và 7 bước với đạo hàm complex-step của (4a), không chép gradient."""
        for iterations in (1, 7):
            _, expected = self.run_bpg(iterations=0)
            for _ in range(iterations):
                gradient = np.empty_like(expected)
                for index in np.ndindex(expected.shape):
                    probe = expected.astype(complex)
                    probe[index] += 1e-25j
                    gradient[index] = objective(self.W.toarray(), probe).imag / 1e-25
                logits = np.log(expected) - .4 * gradient
                expected = np.exp(logits - logits.max(axis=1, keepdims=True))
                expected /= expected.sum(axis=1, keepdims=True)
            _, actual = self.run_bpg(iterations=iterations)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)

    def test_objective_descent(self):
        """f(X) giảm từng vòng với đồ thị này và bước nhỏ; không khẳng định mọi bước học."""
        values = [objective(self.W.toarray(), self.run_bpg(step=.1, iterations=t)[1])
                  for t in range(21)]
        self.assertTrue((np.diff(values) <= 1e-12).all())
        self.assertLess(values[-1], values[0] - 1e-6)

    def test_separated_communities(self):
        """Tách đúng hai thành phần rời nhau trên ví dụ cố định, không phụ thuộc tên nhãn."""
        W = csr_array([[0, 10, 0, 0], [10, 0, 0, 0],
                       [0, 0, 0, 10], [0, 0, 10, 0]], dtype=float)
        R = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
        labels, X = self.run_bpg(W=W, R=R, step=1, iterations=100)
        groups = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(labels[:, None] == labels, groups[:, None] == groups)
        self.assertLess(objective(W.toarray(), X), 1e-6)

    def test_weight_scale_invariance(self):
        """Nhân toàn bộ trọng số với số dương không đổi kết quả của tỉ số NCut."""
        _, expected = self.run_bpg()
        for scale in (1e-6, 7, 1e6):
            _, actual = self.run_bpg(W=self.W * scale)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_permutation_equivariance(self):
        """Đổi thứ tự đỉnh/cột khởi tạo chỉ hoán vị tương ứng xác suất đầu ra."""
        _, expected = self.run_bpg()
        rows, cols = np.array([2, 0, 3, 1]), np.array([2, 0, 1])
        _, actual = self.run_bpg(W=self.W[rows][:, rows], R=self.R[rows][:, cols])
        np.testing.assert_allclose(actual, expected[rows][:, cols], atol=1e-13)

    def test_isolated_vertex(self):
        """Đỉnh cô lập giữ xác suất ban đầu, không ảnh hưởng các đỉnh có cạnh."""
        W = np.zeros((5, 5))
        W[:4, :4] = self.W.toarray()
        R = np.vstack([self.R, [.1, .9, .3]])
        _, expected = self.run_bpg()
        _, actual = self.run_bpg(W=csr_array(W), R=R)
        np.testing.assert_allclose(actual[:4], expected, atol=1e-13)
        initial = 1 / 3 + .1 * R[-1]
        np.testing.assert_allclose(actual[-1], initial / initial.sum())

    def test_input_unchanged_and_repeatable(self):
        """Không sửa dữ liệu/cấu trúc CSR; cùng khởi tạo cho cùng kết quả."""
        before = [a.copy() for a in (self.W.data, self.W.indices, self.W.indptr)]
        _, first = self.run_bpg()
        _, second = self.run_bpg()
        np.testing.assert_array_equal(first, second)
        for actual, expected in zip((self.W.data, self.W.indices, self.W.indptr), before):
            np.testing.assert_array_equal(actual, expected)

    def test_edgeless_graph_stability(self):
        """W=0 cần tránh 0/0 theo lưu ý epsilon mục 3.1, không trả nhãn từ NaN."""
        _, initial = self.run_bpg(iterations=0)
        _, actual = self.run_bpg(W=csr_array((4, 4), dtype=float))
        np.testing.assert_allclose(actual, initial)

    def test_large_step_stability(self):
        """Stress bước học lớn: X hữu hạn; kiểm tra số học, không yêu cầu hội tụ."""
        self.run_bpg(step=1e6, iterations=1)

    def check_ground_truth(self, n, edges, k, known_labels=None, known_score=None,
                           iterations=200, seeds=(0, 1, 42)):
        """So từng seed với vét cạn cùng số cụm thực tế; không bắt buộc BPG tạo đủ K cụm."""
        W = np.zeros((n, n))
        for u, v, weight in edges:
            W[u, v] = W[v, u] = weight
        optimum, truth = exact_partition(W, k)
        if known_score is not None:
            self.assertAlmostEqual(optimum, known_score, places=12)
        if known_labels is not None:
            self.assertAlmostEqual(discrete_ncut(W, np.array(known_labels), k), optimum, places=12)
        self.ground_truth_report = [f'Ground truth: {(truth + 1).tolist()} | NCut = {optimum:.6f}'] #type: ignore
        gaps, optima = [], {k: optimum}
        for seed in seeds:
            R = np.random.default_rng(seed).random((n, k))
            _, initial = self.run_bpg(W=csr_array(W), R=R, iterations=0)
            labels, X = self.run_bpg(W=csr_array(W), R=R, step=1, iterations=iterations)
            score = discrete_ncut(W, labels - 1, k)
            count = len(np.unique(labels))
            if count not in optima:
                optima[count] = exact_partition(W, count)[0]
            reference = optima[count]
            gaps.append(score - reference)
            self.ground_truth_report.append(
                f'Seed {seed:2}: BPG {labels.tolist()} | {count} cụm | NCut = {score:.6f}'
                f' | tối ưu {count} cụm = {reference:.6f} | lệch = {score - reference:.6f}')
            self.assertLess(objective(W, X), objective(W, initial), f'Seed {seed}: f(X) không giảm')
        self.assertTrue(np.allclose(gaps, 0, rtol=0, atol=1e-10),
                        'BPG chưa đạt tối ưu trong ngân sách vòng lặp; có thể chưa hội tụ hoặc mắc cực tiểu địa phương')

    def test_ground_truth_two_triangles(self):
        """Ground truth: hai tam giác cạnh 10 nối bằng cầu 1; tối ưu {1,2,3}/{4,5,6}."""
        self.check_ground_truth(6, [(0, 1, 10), (1, 2, 10), (0, 2, 10),
                                    (3, 4, 10), (4, 5, 10), (3, 5, 10), (2, 3, 1)],
                                2, [0, 0, 0, 1, 1, 1], 1 / 61)

    def test_ground_truth_three_pairs(self):
        """Ground truth: ba cặp cạnh 10 nối vòng bằng cạnh 1; tối ưu {1,2}/{3,4}/{5,6}."""
        self.check_ground_truth(6, [(0, 1, 10), (2, 3, 10), (4, 5, 10),
                                    (1, 2, 1), (3, 4, 1), (5, 0, 1)],
                                3, [0, 0, 1, 1, 2, 2], 3 / 22)

    def test_ground_truth_weighted_path(self):
        """Ground truth: đường đi trọng số 8-1-7; tối ưu {1,2}/{3,4}, volume không đều."""
        self.check_ground_truth(4, [(0, 1, 8), (1, 2, 1), (2, 3, 7)],
                                2, [0, 0, 1, 1], .5 * (1 / 17 + 1 / 15))


    def test_hard_cycle_three_partitions(self):
        """Khó: vòng 8 đỉnh đều, K=3, nhiều nghiệm tối ưu; 5 seed, mỗi seed 1.000 vòng."""
        self.check_ground_truth(8, [(i, (i + 1) % 8, 1) for i in range(8)], 3,
                                [0, 0, 0, 1, 1, 1, 2, 2], 7 / 12,
                                iterations=1000, seeds=(0, 1, 2, 3, 42))

    def test_hard_unbalanced_noisy_clusters(self):
        """Khó: cụm 5-3-2 đỉnh, cạnh chéo mạnh gây nhập/tách cụm; 5 seed x 1.000 vòng."""
        groups = [0] * 5 + [1] * 3 + [2] * 2
        edges = [(i, j, (3, 6, 9)[groups[i]]) for i in range(10) for j in range(i + 1, 10)
                 if groups[i] == groups[j]]
        edges += [(0, 5, 5), (1, 6, 4), (2, 8, 5), (4, 9, 3), (7, 8, 6), (3, 7, 2)]
        self.check_ground_truth(10, edges, 3, iterations=1000, seeds=(0, 1, 2, 3, 42))

    def test_hard_dense_irregular_graph(self):
        """Khó: đồ thị dày 9 đỉnh, trọng số ngẫu nhiên cố định, K=3; 5 seed x 1.000 vòng."""
        weights = np.random.default_rng(2026).integers(1, 10, size=(9, 9))
        edges = [(i, j, int(weights[i, j])) for i in range(9) for j in range(i + 1, 9)]
        self.check_ground_truth(9, edges, 3, iterations=1000, seeds=(0, 1, 2, 3, 42))


class CompactResult(unittest.TestResult):
    """In một trạng thái sau mỗi test, kèm thông báo lỗi ngắn gọn."""
    def show(self, test, status, error=None):
        """Hiển thị số thứ tự, kết quả và mô tả của test."""
        color = '\033[92m' if status == 'PASS' else '\033[91m'
        print(f"{self.testsRun:02d}. [{color}{status:5}\033[0m] {test.shortDescription() or test.id()}", flush=True)
        for line in getattr(test, 'ground_truth_report', []):
            print(f'            {line}', flush=True)
        if error:
            message = ' '.join(str(error[1]).split()) or error[0].__name__
            print(f"            Lý do: {message}", flush=True)

    def addSuccess(self, test):
        """Chỉ in PASS khi test hoàn tất thành công."""
        super().addSuccess(test)
        self.show(test, 'PASS')

    def addFailure(self, test, err):
        """Ghi nhận assertion thất bại và in nguyên nhân."""
        super().addFailure(test, err)
        self.show(test, 'FAIL', err)

    def addError(self, test, err):
        """Phân biệt lỗi thực thi với assertion thất bại."""
        super().addError(test, err)
        self.show(test, 'ERROR', err)


class CompactRunner:
    def run(self, suite):
        """Chạy toàn bộ test, in tổng kết và giữ exit code thất bại nếu có lỗi."""
        print('KIỂM TRA BREGMAN PROXIMAL GRADIENT\n', flush=True)
        result = CompactResult()
        suite.run(result)
        failed, errors = len(result.failures), len(result.errors)
        passed = result.testsRun - failed - errors - len(result.skipped)
        print(f"\nTỔNG KẾT: {result.testsRun} test | \033[92m{passed} PASS\033[0m | "
              f"\033[91m{failed} FAIL\033[0m | {errors} ERROR", flush=True)
        return result


if __name__ == '__main__':
    unittest.main(testRunner=CompactRunner())
