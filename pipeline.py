"""So sánh các cách phân bổ trên cùng dữ liệu; không sửa thuật toán trong main.py."""
import json
from pathlib import Path

import numpy as np

from main import (
    PartitionScheme, CreatePartitionScheme, PartitionScheme_BPG,
    BregmanProximalGradient, BPG_PartitionScheme, FirstPartitioning,
    Partitioning, PartitioningMinimumCutGraph,
)

def create_partition_scheme(
    num_records: int = 100,
    num_transactions: int = 1000,
    seed: int = 42,
    transactions_path: str | Path | None = None,
) -> PartitionScheme:
    """Tạo scheme ban đầu bằng class CreatePartitionScheme trong main.py."""
    creator = CreatePartitionScheme(num_records=num_records, seed=seed)
    creator.generate_mock_data()
    creator.generate_mock_transactions(num_transactions=num_transactions)
    if transactions_path is not None:
        Path(transactions_path).write_text(
            json.dumps(creator.transactions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return creator.run()


if __name__ == "__main__":
    transactions_path = Path(__file__).with_name("mock_transactions.json")
    partition_scheme = create_partition_scheme(
        num_records=100,
        num_transactions=1000,
        seed=42,
        transactions_path=transactions_path,
    )
    print("Mock transactions:", transactions_path.name)
    partition_scheme = FirstPartitioning(partition_scheme, site_num=5).run()
    print("Heat graph:", partition_scheme.heat_graph.number_of_nodes(), "nodes,",
          partition_scheme.heat_graph.number_of_edges(), "edges")
    print("Partitions:")
    for partition_id, vertices in partition_scheme.partitions.items():
        print(f"Partition {partition_id}: {sorted(vertices)}")

    bpg_scheme = PartitionScheme_BPG(partition_scheme, use_coarsening=True)
    W = bpg_scheme.run()
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        labels = BregmanProximalGradient(
            W, partitions_num=5, step_size=1, iterations_num=1000,
        ).run()
    bpg_partition_scheme = BPG_PartitionScheme(
        partition_scheme.heat_graph, labels, bpg_scheme.coarse_to_nodes,
    ).run()

    print("\nPartitions after BPG:")
    for partition_id, vertices in bpg_partition_scheme.partitions.items():
        print(f"Partition {partition_id}: {sorted(vertices)}")

    # So sánh nhóm đỉnh, không phụ thuộc vào ID partition (0-4 và 1-5).
    before_groups = {frozenset(v) for v in partition_scheme.partitions.values()}
    after_groups = {frozenset(v) for v in bpg_partition_scheme.partitions.values()}
    print("\nPartition groups changed:", before_groups != after_groups)
    for name, scheme in [("Before BPG", partition_scheme), ("After BPG", bpg_partition_scheme)]:
        calculator = PartitioningMinimumCutGraph(scheme, workload_factor=0.1)
        loads = {
            pid: sum(scheme.heat_graph.nodes[v]["weight"] for v in vertices)
            for pid, vertices in scheme.partitions.items()
        }
        print(f"{name}: partition sizes = { {p: len(v) for p, v in scheme.partitions.items()} }")
        print(f"{name}: workloads = {loads}")
        print(f"{name}: cut weight = {calculator.calculateMinimumCut(scheme):g}")

