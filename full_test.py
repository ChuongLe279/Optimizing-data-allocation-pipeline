"""Chạy toàn bộ test: python -B full_test.py. Không ghi cache hay tạo file kết quả."""
import sys
import unittest

sys.dont_write_bytecode = True
from test_unbalanced_graph import CompactRunner


def load_tests(loader, tests, pattern):
    """Nạp đúng năm bộ test một lần, không dùng discovery kéo thêm file ngoài phạm vi."""
    return unittest.TestSuite(loader.loadTestsFromName(name) for name in (
        'test_bpg', 'test_unbalanced_graph', 'test_min_cut_graph', 'test_first_partitioning', 'test_partitioning',
    ))


if __name__ == '__main__':
    unittest.main(testRunner=CompactRunner('TOÀN BỘ THUẬT TOÁN', grouped=True))
