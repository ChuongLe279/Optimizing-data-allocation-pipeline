import unittest
import numpy as np
from scipy.sparse import csr_array
from main import BregmanProximalGradient

class TestBPG(unittest.TestCase):
    def test_bpg_output(self):
        W = csr_array([
            [0, 10, 1, 0],
            [10, 0, 0, 1],
            [1, 0, 0, 10],
            [0, 1, 10, 0]
        ], dtype=float)

        assert W.shape is not None
        N = W.shape[0]
        K = 2

        bpg = BregmanProximalGradient(
            W=W,
            partitions_num=K,
            step_size=0.1,
            iterations_num=100
        )

        labels = bpg.run()

        assert isinstance(labels, np.ndarray), "Đầu ra phải là numpy array"
        assert labels.shape == (N,), f"Đầu ra phải có dạng ({N},)"
        assert np.issubdtype(labels.dtype, np.integer), "Nhãn phải là số nguyên"
        assert np.all(labels >= 1), "Nhãn phải >= 1"
        assert np.all(labels <= K), f"Nhãn phải <= {K}"

        print("Test đầu ra: PASS")
        print("Labels:", labels)

if __name__ == "__main__":
    test = TestBPG()
    test.test_bpg_output()