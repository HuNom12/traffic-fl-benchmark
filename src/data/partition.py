
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "METR-LA"
TRAIN_X_PATH = PROCESSED_DIR / "train" / "X_train.npy"
ADJACENCY_PATH = PROCESSED_DIR / "graph" / "adjacency_matrix.npy"

SPLITS_DIR = PROJECT_ROOT / "splits"

SPATIAL_OUTPUT_PATH = SPLITS_DIR / "spatial_skew.json"
TEMPORAL_OUTPUT_PATH = SPLITS_DIR / "temporal_shift.json"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# UTILS
# ============================================================

def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Lưu cấu hình partition thành JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    logger.info("Saved: %s", output_path)


def validate_inputs(
    X: np.ndarray,
    adjacency: np.ndarray,
    num_nodes: int,
) -> None:
    """Kiểm tra dữ liệu đầu vào."""

    if X.ndim != 4:
        raise ValueError(
            f"X phải có 4 chiều (B,T,N,F), nhận được {X.shape}"
        )

    if X.shape[2] != num_nodes:
        raise ValueError(
            f"X có {X.shape[2]} nodes, yêu cầu {num_nodes}"
        )

    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Adjacency phải có shape "
            f"({num_nodes}, {num_nodes}), nhận {adjacency.shape}"
        )

    if not np.isfinite(adjacency).all():
        raise ValueError("Adjacency chứa NaN hoặc Inf")

    if not np.isfinite(X).all():
        raise ValueError("X chứa NaN hoặc Inf")


# ============================================================
# K-MEANS IMPLEMENTATION
# ============================================================

def kmeans(
    data: np.ndarray,
    num_clusters: int,
    seed: int = 42,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    K-Means đơn giản, không cần scikit-learn.

    Parameters
    ----------
    data:
        Shape (N, D), N samples và D features.
    num_clusters:
        Số clusters.
    seed:
        Random seed để tái lập kết quả.

    Returns
    -------
    labels:
        Cluster ID của từng sample.
    centroids:
        Tâm của các clusters.
    """

    n_samples = data.shape[0]

    if not 1 <= num_clusters <= n_samples:
        raise ValueError("num_clusters phải nằm trong [1, n_samples]")

    rng = np.random.default_rng(seed)

    # Khởi tạo centroids bằng K-Means++
    centroids = np.empty(
        (num_clusters, data.shape[1]),
        dtype=np.float64,
    )

    first_idx = rng.integers(n_samples)
    centroids[0] = data[first_idx]

    min_distances = np.sum(
        (data - centroids[0]) ** 2,
        axis=1,
    )

    for cluster_id in range(1, num_clusters):
        total_distance = min_distances.sum()

        if total_distance <= 1e-12:
            selected_idx = rng.integers(n_samples)
        else:
            probabilities = min_distances / total_distance
            selected_idx = rng.choice(
                n_samples,
                p=probabilities,
            )

        centroids[cluster_id] = data[selected_idx]

        distances = np.sum(
            (data - centroids[cluster_id]) ** 2,
            axis=1,
        )

        min_distances = np.minimum(min_distances, distances)

    labels = np.zeros(n_samples, dtype=np.int64)

    for _ in range(max_iter):
        distances = np.sum(
            (data[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )

        new_labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()

        for cluster_id in range(num_clusters):
            members = data[new_labels == cluster_id]

            if len(members) > 0:
                new_centroids[cluster_id] = members.mean(axis=0)
            else:
                # Tránh cluster rỗng
                new_centroids[cluster_id] = data[
                    rng.integers(n_samples)
                ]

        if np.array_equal(labels, new_labels):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids


# ============================================================
# SPATIAL FEATURE EXTRACTION
# ============================================================

def extract_node_features(X: np.ndarray) -> np.ndarray:
    """
    Tạo vector đặc trưng cho từng node từ speed channel.

    X: (B, T, N, F)

    Mỗi node được mô tả bởi:
    - Mean speed
    - Std speed
    - 25th percentile
    - Median speed
    - 75th percentile
    """

    num_nodes = X.shape[2]
    features = []

    for node_id in range(num_nodes):
        # Chỉ sử dụng speed channel (feature 0)
        speed = np.asarray(
            X[:, :, node_id, 0],
            dtype=np.float64,
        ).reshape(-1)

        node_feature = [
            np.mean(speed),
            np.std(speed),
            np.percentile(speed, 25),
            np.percentile(speed, 50),
            np.percentile(speed, 75),
        ]

        features.append(node_feature)

    return np.asarray(features, dtype=np.float64)


def extract_graph_embedding(
    adjacency: np.ndarray,
    num_components: int,
) -> np.ndarray:
    """
    Trích xuất graph embedding bằng phổ của normalized adjacency.

    A đã là ma trận trọng số Gaussian, không tính Gaussian Kernel lại.

    Sử dụng A_sym = (A + A.T) / 2 để tạo đồ thị vô hướng
    phục vụ clustering.
    """

    A = np.maximum(adjacency, adjacency.T).astype(np.float64)

    # Loại bỏ self-loop khi tính graph embedding
    np.fill_diagonal(A, 0.0)

    degree = A.sum(axis=1)
    degree = np.maximum(degree, 1e-12)

    D_inv_sqrt = 1.0 / np.sqrt(degree)

    normalized_A = (
        D_inv_sqrt[:, None]
        * A
        * D_inv_sqrt[None, :]
    )

    # Vì normalized_A đối xứng, dùng eigh
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_A)

    # Lấy các eigenvectors ứng với eigenvalues lớn nhất
    selected_indices = np.argsort(eigenvalues)[::-1][
        :num_components
    ]

    embedding = eigenvectors[:, selected_indices]

    return embedding


def standardize_features(features: np.ndarray) -> np.ndarray:
    """Z-score từng cột đặc trưng."""

    mean = features.mean(axis=0)
    std = features.std(axis=0)

    std = np.where(std < 1e-8, 1.0, std)

    return (features - mean) / std


# ============================================================
# SPATIAL SKEW
# ============================================================

def generate_spatial_skew(
    X: np.ndarray,
    adjacency: np.ndarray,
    num_clients: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Phân cụm 207 nodes thành num_clients clusters.

    Kết hợp:
    1. Đặc trưng vận tốc của nodes.
    2. Graph embedding từ adjacency matrix.

    Mỗi client nhận một cluster nodes.
    """

    num_nodes = X.shape[2]

    if num_clients > num_nodes:
        raise ValueError("num_clients không thể lớn hơn num_nodes")

    logger.info("Generating spatial skew...")
    logger.info("Number of nodes: %d", num_nodes)
    logger.info("Number of clients: %d", num_clients)

    # 1. Traffic features
    traffic_features = extract_node_features(X)

    # 2. Graph features
    graph_features = extract_graph_embedding(
        adjacency,
        num_components=min(num_clients, num_nodes),
    )

    # 3. Chuẩn hóa từng nhóm feature
    traffic_features = standardize_features(traffic_features)
    graph_features = standardize_features(graph_features)

    # 4. Kết hợp traffic + graph
    #
    # Traffic có trọng số lớn hơn để đảm bảo
    # clusters có đặc trưng vận tốc khác nhau.
    combined_features = np.concatenate(
        [
            traffic_features,
            0.5 * graph_features,
        ],
        axis=1,
    )

    # 5. Clustering
    labels, centroids = kmeans(
        combined_features,
        num_clusters=num_clients,
        seed=seed,
    )

    # 6. Tạo node index cho từng client
    clients = {}

    for client_id in range(num_clients):
        node_indices = np.where(labels == client_id)[0]

        clients[f"client_{client_id}"] = {
            "node_indices": node_indices.tolist(),
            "num_nodes": int(len(node_indices)),
            "cluster_id": client_id,
        }

    # Kiểm tra mỗi node chỉ thuộc một client
    all_nodes = [
        node
        for client in clients.values()
        for node in client["node_indices"]
    ]

    if len(all_nodes) != num_nodes:
        raise RuntimeError("Số lượng node sau clustering không hợp lệ")

    if len(set(all_nodes)) != num_nodes:
        raise RuntimeError("Có node bị trùng giữa các clients")

    if set(all_nodes) != set(range(num_nodes)):
        raise RuntimeError("Có node bị thiếu sau clustering")

    result = {
        "scenario": "spatial_skew",
        "dataset": "METR-LA",
        "num_nodes": num_nodes,
        "num_clients": num_clients,
        "seed": seed,
        "method": {
            "clustering": "KMeans",
            "traffic_features": [
                "speed_mean",
                "speed_std",
                "speed_p25",
                "speed_median",
                "speed_p75",
            ],
            "graph_features": "normalized_adjacency_spectral_embedding",
            "graph_fusion_weight": 0.5,
            "adjacency_symmetrization": "maximum(A, A.T)",
        },
        "node_order": "same_as_X_train_and_adjacency",
        "clients": clients,
    }

    logger.info("Spatial clustering completed.")

    for client_id, client in clients.items():
        logger.info(
            "%s: %d nodes",
            client_id,
            client["num_nodes"],
        )

    return result


# ============================================================
# TEMPORAL SHIFT
# ============================================================

def allocate_indices(
    indices: np.ndarray,
    weights: list[float],
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Phân bổ một nhóm sample cho clients theo tỷ lệ weights.

    Mỗi sample chỉ được gán cho đúng một client.
    """

    num_clients = len(weights)

    if len(indices) == 0:
        return [[] for _ in range(num_clients)]

    weights_array = np.asarray(weights, dtype=np.float64)
    weights_array = weights_array / weights_array.sum()

    shuffled = rng.permutation(indices)

    raw_counts = weights_array * len(shuffled)
    counts = np.floor(raw_counts).astype(int)

    # Phân bổ phần dư theo phương pháp largest remainder
    remainder = len(shuffled) - counts.sum()

    fractional_parts = raw_counts - counts
    order = np.argsort(fractional_parts)[::-1]

    for i in range(remainder):
        counts[order[i]] += 1

    result = []
    start = 0

    for count in counts:
        end = start + count
        client_indices = shuffled[start:end]

        result.append(
            np.sort(client_indices).astype(int).tolist()
        )

        start = end

    return result


def generate_temporal_shift(
    X: np.ndarray,
    num_clients: int = 5,
    seed: int = 42,
    morning_peak: tuple[float, float] = (7.0, 10.0),
    evening_peak: tuple[float, float] = (16.0, 19.0),
) -> dict[str, Any]:
    """
    Chia các sample theo thời điểm cao điểm và thấp điểm.

    X[..., 1] là time_of_day, được biểu diễn dưới dạng
    tỷ lệ trong ngày [0, 1].

    Peak:
        07:00 - 10:00
        16:00 - 19:00

    Off-peak:
        Các thời điểm còn lại.

    Các clients có tỷ lệ peak/off-peak khác nhau.
    """

    num_samples = X.shape[0]
    num_nodes = X.shape[2]

    if num_clients < 2:
        raise ValueError("Temporal shift cần ít nhất 2 clients")

    logger.info("Generating temporal shift...")
    logger.info("Number of samples: %d", num_samples)

    # Lấy time_of_day tại timestep cuối của X.
    # Các node trong cùng một timestamp thường có cùng time_of_day.
    time_values = np.asarray(
        X[:, -1, :, 1],
        dtype=np.float64,
    )

    sample_time = np.median(time_values, axis=1)

    if not np.isfinite(sample_time).all():
        raise ValueError("time_of_day chứa NaN hoặc Inf")

    if sample_time.min() < -1e-5 or sample_time.max() > 1.00001:
        raise ValueError(
            "time_of_day cần nằm trong [0, 1]. "
            "Hãy kiểm tra cách mã hóa time_of_day."
        )

    sample_time = np.clip(sample_time, 0.0, 1.0)

    # Chuyển tỷ lệ trong ngày thành giờ
    sample_hour = sample_time * 24.0

    def in_hour_range(
        hours: np.ndarray,
        time_range: tuple[float, float],
    ) -> np.ndarray:
        start, end = time_range

        if start <= end:
            return (hours >= start) & (hours < end)

        # Hỗ trợ khoảng thời gian qua nửa đêm
        return (hours >= start) | (hours < end)

    morning_mask = in_hour_range(sample_hour, morning_peak)
    evening_mask = in_hour_range(sample_hour, evening_peak)

    peak_mask = morning_mask | evening_mask
    offpeak_mask = ~peak_mask

    peak_indices = np.where(peak_mask)[0]
    offpeak_indices = np.where(offpeak_mask)[0]

    logger.info("Peak samples: %d", len(peak_indices))
    logger.info("Off-peak samples: %d", len(offpeak_indices))

    # Client 0 thiên về peak, client cuối thiên về off-peak.
    #
    # Ví dụ 5 clients:
    # Peak:    [0.60, 0.50, 0.30, 0.15, 0.10]
    # Offpeak: [0.10, 0.15, 0.30, 0.50, 0.60]
    #
    # Mỗi nhóm vẫn được phân bổ cho tất cả clients,
    # nhưng tỷ lệ phân bổ khác nhau.
    if num_clients == 2:
        peak_weights = [0.8, 0.2]
    else:
        peak_weights = np.linspace(
            0.6,
            0.1,
            num_clients,
        ).tolist()

    offpeak_weights = peak_weights[::-1]

    rng = np.random.default_rng(seed)

    peak_allocations = allocate_indices(
        peak_indices,
        peak_weights,
        rng,
    )

    offpeak_allocations = allocate_indices(
        offpeak_indices,
        offpeak_weights,
        rng,
    )

    clients = {}

    for client_id in range(num_clients):
        indices = sorted(
            peak_allocations[client_id]
            + offpeak_allocations[client_id]
        )

        client_peak_count = len(peak_allocations[client_id])
        client_offpeak_count = len(offpeak_allocations[client_id])
        total = len(indices)

        clients[f"client_{client_id}"] = {
            "sample_indices": indices,
            "num_samples": total,
            "peak_samples": client_peak_count,
            "offpeak_samples": client_offpeak_count,
            "peak_ratio": (
                client_peak_count / total if total > 0 else 0.0
            ),
        }

    # Kiểm tra mọi sample được phân bổ đúng một lần
    all_indices = [
        index
        for client in clients.values()
        for index in client["sample_indices"]
    ]

    if len(all_indices) != num_samples:
        raise RuntimeError(
            "Số sample sau temporal partition không hợp lệ"
        )

    if len(set(all_indices)) != num_samples:
        raise RuntimeError(
            "Có sample bị trùng giữa các clients"
        )

    if set(all_indices) != set(range(num_samples)):
        raise RuntimeError(
            "Có sample bị thiếu sau temporal partition"
        )

    result = {
        "scenario": "temporal_shift",
        "dataset": "METR-LA",
        "num_samples": num_samples,
        "num_nodes": num_nodes,
        "num_clients": num_clients,
        "seed": seed,
        "time_feature": "time_of_day",
        "time_encoding": "fraction_of_day",
        "peak_hours": {
            "morning": list(morning_peak),
            "evening": list(evening_peak),
        },
        "offpeak_definition": "all_remaining_hours",
        "allocation": {
            "peak_weights": peak_weights,
            "offpeak_weights": offpeak_weights,
            "method": "weighted_random_disjoint_partition",
        },
        "clients": clients,
    }

    logger.info("Temporal partition completed.")

    for client_id, client in clients.items():
        logger.info(
            "%s: samples=%d, peak_ratio=%.4f",
            client_id,
            client["num_samples"],
            client["peak_ratio"],
        )

    return result


# ============================================================
# PIPELINE
# ============================================================

def generate_partitions(
    num_clients: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tạo và lưu hai kịch bản Non-IID."""

    if not TRAIN_X_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy X_train: {TRAIN_X_PATH}"
        )

    if not ADJACENCY_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy adjacency: {ADJACENCY_PATH}"
        )

    logger.info("Loading METR-LA data...")

    # mmap để tránh nạp toàn bộ X_train vào RAM
    X_train = np.load(
        TRAIN_X_PATH,
        mmap_mode="r",
    )

    adjacency = np.load(ADJACENCY_PATH)

    validate_inputs(
        X_train,
        adjacency,
        num_nodes=207,
    )

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("Adjacency shape: %s", adjacency.shape)

    # Spatial skew
    spatial_result = generate_spatial_skew(
        X=X_train,
        adjacency=adjacency,
        num_clients=num_clients,
        seed=seed,
    )

    # Temporal shift
    temporal_result = generate_temporal_shift(
        X=X_train,
        num_clients=num_clients,
        seed=seed,
    )

    # Save to repository root/splits/
    save_json(
        spatial_result,
        SPATIAL_OUTPUT_PATH,
    )

    save_json(
        temporal_result,
        TEMPORAL_OUTPUT_PATH,
    )

    logger.info("All partition files generated successfully.")

    return spatial_result, temporal_result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Non-IID partitions for METR-LA"
    )

    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of federated clients",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    generate_partitions(
        num_clients=args.num_clients,
        seed=args.seed,
    )
