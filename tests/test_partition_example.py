import json
import numpy as np
import pytest

from src.data import partition
from src.data.partition import (
    generate_spatial_skew,
    generate_temporal_shift,
    generate_partitions,
)


def _create_mock_traffic_data(num_samples: int = 120, num_nodes: int = 50, seed: int = 42):
    """Tạo tensor 4D giả lập chuẩn METR-LA (B, 12, N, 2) và ma trận kề (N, N)."""
    rng = np.random.default_rng(seed)

    # Feature 0: tốc độ chuẩn hóa; Feature 1: time-of-day nằm trong [0, 1]
    speed = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 12, num_nodes, 1)).astype(np.float32)
    tod = np.linspace(0.0, 1.0, num_samples, endpoint=False, dtype=np.float32)
    tod_grid = np.broadcast_to(tod[:, None, None, None], (num_samples, 12, num_nodes, 1)).copy()
    X = np.concatenate([speed, tod_grid], axis=-1)

    # Ma trận kề đối xứng ngẫu nhiên có đường chéo bằng 1
    raw_adj = rng.uniform(0.0, 1.0, size=(num_nodes, num_nodes)).astype(np.float32)
    adj = (raw_adj + raw_adj.T) / 2.0
    np.fill_diagonal(adj, 1.0)
    return X, adj


def test_spatial_skew_partition_example():
    """Kiểm tra phân cụm không gian (Spatial Skew) chia đủ N nodes cho K clients không trùng lặp."""
    X, adj = _create_mock_traffic_data(num_samples=120, num_nodes=50, seed=42)
    num_clients = 5

    result = generate_spatial_skew(X=X, adjacency=adj, num_clients=num_clients, seed=42)

    assert result["scenario"] == "spatial_skew"
    assert result["num_clients"] == num_clients
    clients = result["clients"]
    assert len(clients) == num_clients

    all_assigned_nodes = []
    for client_key, client_data in clients.items():
        node_indices = client_data["node_indices"]
        assert len(node_indices) == client_data["num_nodes"]
        assert len(node_indices) > 0, f"{client_key} không được gán node nào"
        all_assigned_nodes.extend(node_indices)

    assert sorted(all_assigned_nodes) == list(range(50))


def test_temporal_shift_partition_example():
    """Kiểm tra phân chia lệch pha thời gian (Temporal Shift) chia đủ B mẫu cho K clients."""
    X, _ = _create_mock_traffic_data(num_samples=120, num_nodes=50, seed=42)
    num_clients = 5

    result = generate_temporal_shift(X=X, num_clients=num_clients, seed=42)

    assert result["scenario"] == "temporal_shift"
    assert result["num_clients"] == num_clients
    clients = result["clients"]
    assert len(clients) == num_clients

    all_assigned_samples = []
    for client_key, client_data in clients.items():
        sample_indices = client_data["sample_indices"]
        assert len(sample_indices) == client_data["num_samples"]
        assert len(sample_indices) > 0, f"{client_key} không có mẫu thời gian nào"
        all_assigned_samples.extend(sample_indices)

    assert sorted(all_assigned_samples) == list(range(120))


def test_generate_partitions_end_to_end(tmp_path, monkeypatch):
    """Kiểm tra generate_partitions xuất đúng 2 tệp JSON vào thư mục tạm với N=207."""
    X, adj = _create_mock_traffic_data(num_samples=120, num_nodes=207, seed=42)

    mock_x_path = tmp_path / "X_train.npy"
    mock_adj_path = tmp_path / "adjacency_matrix.npy"
    np.save(mock_x_path, X)
    np.save(mock_adj_path, adj)

    spatial_out = tmp_path / "splits" / "spatial_skew.json"
    temporal_out = tmp_path / "splits" / "temporal_shift.json"

    monkeypatch.setattr(partition, "TRAIN_X_PATH", mock_x_path)
    monkeypatch.setattr(partition, "ADJACENCY_PATH", mock_adj_path)
    monkeypatch.setattr(partition, "SPATIAL_OUTPUT_PATH", spatial_out)
    monkeypatch.setattr(partition, "TEMPORAL_OUTPUT_PATH", temporal_out)

    generate_partitions(num_clients=5, seed=42)

    assert spatial_out.exists()
    assert temporal_out.exists()

    with spatial_out.open("r", encoding="utf-8") as f:
        assert len(json.load(f)["clients"]) == 5
    with temporal_out.open("r", encoding="utf-8") as f:
        assert len(json.load(f)["clients"]) == 5