
import json
from pathlib import Path

import numpy as np
import pytest


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "splits"
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "METR-LA"

SPATIAL_PATH = SPLITS_DIR / "spatial_skew.json"
TEMPORAL_PATH = SPLITS_DIR / "temporal_shift.json"

X_TRAIN_PATH = DATA_DIR / "train" / "X_train.npy"
Y_TRAIN_PATH = DATA_DIR / "train" / "Y_train.npy"
ADJACENCY_PATH = DATA_DIR / "graph" / "adjacency_matrix.npy"

NUM_NODES = 207


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def spatial_config():
    assert SPATIAL_PATH.exists(), (
        f"Không tìm thấy {SPATIAL_PATH}"
    )

    with SPATIAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def temporal_config():
    assert TEMPORAL_PATH.exists(), (
        f"Không tìm thấy {TEMPORAL_PATH}"
    )

    with TEMPORAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def train_data():
    assert X_TRAIN_PATH.exists()
    assert Y_TRAIN_PATH.exists()

    X = np.load(X_TRAIN_PATH, mmap_mode="r")
    Y = np.load(Y_TRAIN_PATH, mmap_mode="r")

    return X, Y


@pytest.fixture(scope="module")
def adjacency():
    assert ADJACENCY_PATH.exists()

    return np.load(ADJACENCY_PATH)


# ============================================================
# 1. JSON STRUCTURE
# ============================================================

def test_spatial_json_structure(spatial_config):
    assert spatial_config["scenario"] == "spatial_skew"
    assert spatial_config["dataset"] == "METR-LA"
    assert spatial_config["num_nodes"] == NUM_NODES

    clients = spatial_config["clients"]

    assert isinstance(clients, dict)
    assert len(clients) == spatial_config["num_clients"]

    for client_id, client in clients.items():
        assert "node_indices" in client
        assert "num_nodes" in client
        assert "cluster_id" in client

        assert isinstance(client["node_indices"], list)
        assert client["num_nodes"] == len(client["node_indices"])

    print("\nPASS: Spatial JSON structure")


def test_temporal_json_structure(temporal_config):
    assert temporal_config["scenario"] == "temporal_shift"
    assert temporal_config["dataset"] == "METR-LA"

    clients = temporal_config["clients"]

    assert isinstance(clients, dict)
    assert len(clients) == temporal_config["num_clients"]

    for client_id, client in clients.items():
        assert "sample_indices" in client
        assert "num_samples" in client
        assert "peak_samples" in client
        assert "offpeak_samples" in client
        assert "peak_ratio" in client

        assert isinstance(client["sample_indices"], list)
        assert client["num_samples"] == len(client["sample_indices"])

    print("\nPASS: Temporal JSON structure")


# ============================================================
# 2. SPATIAL NODE VALIDATION
# ============================================================

def test_spatial_node_range(spatial_config):
    clients = spatial_config["clients"]

    all_nodes = []

    for client in clients.values():
        nodes = client["node_indices"]

        assert all(isinstance(n, int) for n in nodes)
        assert all(0 <= n < NUM_NODES for n in nodes)

        all_nodes.extend(nodes)

    assert len(all_nodes) == NUM_NODES
    assert len(set(all_nodes)) == NUM_NODES

    assert set(all_nodes) == set(range(NUM_NODES))

    print("\nPASS: All 207 nodes are allocated exactly once")


def test_spatial_client_distribution(spatial_config):
    clients = spatial_config["clients"]

    node_counts = [
        client["num_nodes"]
        for client in clients.values()
    ]

    assert all(count > 0 for count in node_counts)

    print("\nSpatial node distribution:")

    for client_id, client in clients.items():
        print(
            f"{client_id}: "
            f"{client['num_nodes']} nodes"
        )

    print("PASS: Every client has nodes")


def test_spatial_adjacency_compatibility(
    spatial_config,
    adjacency,
):
    assert adjacency.shape == (NUM_NODES, NUM_NODES)

    for client_id, client in spatial_config["clients"].items():
        nodes = client["node_indices"]

        # Lấy adjacency submatrix của client
        client_adjacency = adjacency[np.ix_(nodes, nodes)]

        assert client_adjacency.shape == (
            len(nodes),
            len(nodes),
        )

        assert np.isfinite(client_adjacency).all()

    print("\nPASS: Spatial adjacency submatrices are valid")


# ============================================================
# 3. SPATIAL NON-IID VALIDATION
# ============================================================

def test_spatial_speed_distribution(
    spatial_config,
    train_data,
):
    X, _ = train_data

    client_means = []

    print("\nSpatial speed distribution:")

    for client_id, client in spatial_config["clients"].items():
        nodes = client["node_indices"]

        # Speed channel = feature 0
        speed = np.asarray(X[:, :, nodes, 0])

        mean_speed = float(speed.mean())
        std_speed = float(speed.std())

        client_means.append(mean_speed)

        print(
            f"{client_id}: "
            f"mean={mean_speed:.4f}, "
            f"std={std_speed:.4f}"
        )

    # Kiểm tra các client không có cùng đặc trưng speed
    assert np.std(client_means) > 1e-6, (
        "Các clients có phân phối speed gần như giống nhau"
    )

    print("PASS: Spatial speed distributions differ")


# ============================================================
# 4. TEMPORAL SAMPLE VALIDATION
# ============================================================

def test_temporal_sample_range(
    temporal_config,
    train_data,
):
    X, Y = train_data

    num_samples = X.shape[0]

    assert temporal_config["num_samples"] == num_samples
    assert Y.shape[0] == num_samples

    all_indices = []

    for client in temporal_config["clients"].values():
        indices = client["sample_indices"]

        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < num_samples for i in indices)

        all_indices.extend(indices)

    assert len(all_indices) == num_samples
    assert len(set(all_indices)) == num_samples

    assert set(all_indices) == set(range(num_samples))

    print(
        f"\nPASS: All {num_samples} samples "
        "are allocated exactly once"
    )


def test_temporal_client_distribution(temporal_config):
    clients = temporal_config["clients"]

    print("\nTemporal sample distribution:")

    for client_id, client in clients.items():
        total = client["num_samples"]

        assert total > 0

        assert (
            client["peak_samples"]
            + client["offpeak_samples"]
            == total
        )

        expected_ratio = client["peak_samples"] / total

        assert np.isclose(
            client["peak_ratio"],
            expected_ratio,
            atol=1e-6,
        )

        print(
            f"{client_id}: "
            f"samples={total}, "
            f"peak={client['peak_samples']}, "
            f"offpeak={client['offpeak_samples']}, "
            f"peak_ratio={client['peak_ratio']:.4f}"
        )

    print("PASS: Temporal sample statistics are valid")


# ============================================================
# 5. TEMPORAL NON-IID VALIDATION
# ============================================================

def test_temporal_peak_distribution(
    temporal_config,
    train_data,
):
    X, _ = train_data

    morning = temporal_config["peak_hours"]["morning"]
    evening = temporal_config["peak_hours"]["evening"]

    # Lấy time_of_day tại timestep cuối
    time_values = np.asarray(
        X[:, -1, :, 1],
        dtype=np.float64,
    )

    sample_hours = np.median(time_values, axis=1) * 24.0

    def in_range(hours, time_range):
        start, end = time_range

        if start <= end:
            return (hours >= start) & (hours < end)

        return (hours >= start) | (hours < end)

    peak_mask = (
        in_range(sample_hours, morning)
        | in_range(sample_hours, evening)
    )

    client_ratios = []

    print("\nTemporal peak distribution:")

    for client_id, client in temporal_config["clients"].items():
        indices = np.asarray(
            client["sample_indices"],
            dtype=np.int64,
        )

        actual_peak = int(peak_mask[indices].sum())
        actual_offpeak = len(indices) - actual_peak

        actual_ratio = (
            actual_peak / len(indices)
            if len(indices) > 0
            else 0.0
        )

        assert actual_peak == client["peak_samples"]
        assert actual_offpeak == client["offpeak_samples"]

        assert np.isclose(
            actual_ratio,
            client["peak_ratio"],
            atol=1e-6,
        )

        client_ratios.append(actual_ratio)

        print(
            f"{client_id}: "
            f"peak_ratio={actual_ratio:.4f}"
        )

    # Phải có sự khác biệt về tỷ lệ peak giữa các clients
    assert np.ptp(client_ratios) > 0.05, (
        "Temporal partition chưa tạo ra sự khác biệt "
        "đủ rõ về tỷ lệ peak giữa các clients"
    )

    print("PASS: Temporal peak distributions differ")


# ============================================================
# 6. GRAPH AND DATA COMPATIBILITY
# ============================================================

def test_data_shapes(train_data, adjacency):
    X, Y = train_data

    assert X.ndim == 4
    assert Y.ndim == 4

    assert X.shape[2] == NUM_NODES
    assert Y.shape[2] == NUM_NODES

    assert X.shape[1:] == (12, NUM_NODES, 2)
    assert Y.shape[1:] == (12, NUM_NODES, 1)

    assert adjacency.shape == (NUM_NODES, NUM_NODES)

    print("\nPASS: Dataset and graph dimensions are compatible")
