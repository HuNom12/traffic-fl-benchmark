import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.data.dataset import compute_adjacency_matrix, load_and_preprocess_pems
from src.data.partition import (
    partition_nodes_spatial_clustering,
    generate_spatial_skew_scenario,
    generate_temporal_shift_scenario,
)


def test_partition_pipeline():
    print("=== [Test Task 1.3] Non-IID Partition Scenarios ===")

    num_nodes = 20
    num_clients = 4
    adj = compute_adjacency_matrix("data/raw/distance.csv", num_nodes=num_nodes)
    data_dict = load_and_preprocess_pems("data/raw/pems04.npz", time_in=12, time_out=12)
    tx, ty = data_dict["train_x"], data_dict["train_y"]

    # 1. Gom cụm
    clusters = partition_nodes_spatial_clustering(adj, num_clients=num_clients)
    assert len(clusters) == num_clients, "Số lượng cụm không khớp num_clients!"
    total_assigned_nodes = sum(len(nodes) for nodes in clusters.values())
    assert total_assigned_nodes == num_nodes, "Thiếu hoặc trùng lặp sensor khi gom cụm!"
    print(f"-> Spectral Clustering chia {num_nodes} nodes thành {num_clients} clusters [PASSED]")

    # 2. Sinh kịch bản Spatial Skew
    spatial_path = "splits/spatial_skew.json"
    generate_spatial_skew_scenario(tx, ty, clusters, output_path=spatial_path)
    assert os.path.exists(spatial_path), f"Không tìm thấy file {spatial_path}!"
    with open(spatial_path, "r", encoding="utf-8") as f:
        spatial_data = json.load(f)
        assert len(spatial_data["clients"]) == num_clients
    print(f"-> Kịch bản Spatial Skew (lưu tại {spatial_path}) [PASSED]")

    # 3. Sinh kịch bản Temporal Shift
    temporal_path = "splits/temporal_shift.json"
    generate_temporal_shift_scenario(tx, num_clients=num_clients, output_path=temporal_path)
    assert os.path.exists(temporal_path), f"Không tìm thấy file {temporal_path}!"
    with open(temporal_path, "r", encoding="utf-8") as f:
        temporal_data = json.load(f)
        assert len(temporal_data["clients"]) == num_clients
    print(f"-> Kịch bản Temporal Shift (lưu tại {temporal_path}) [PASSED]")
    print(">>> TASK 1.3 PASS HOÀN TOÀN <<<\n")


if __name__ == "__main__":
    test_partition_pipeline()