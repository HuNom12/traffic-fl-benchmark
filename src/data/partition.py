import os
import json
import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import SpectralClustering

def partition_nodes_spatial_clustering(
    adj_matrix: np.ndarray,
    num_clients: int = 5
) -> Dict[int, List[int]]:
    """Phân chia N cảm biến về K Client/Node dựa trên đồ thị không gian.

    TODO:
    - Áp dụng Spectral Clustering / Louvain hoặc gom theo tọa độ.
    
    Returns:
        Dict: {client_id: [danh_sách_sensor_id]}
    """
    # TODO: Cài đặt thuật toán gom cụm tại đây
    num_nodes = adj_matrix.shape[0]

    # Đảm bảo ma trận kề đối xứng và không có giá trị âm
    affinity = (adj_matrix + adj_matrix.T) / 2.0
    affinity = np.clip(affinity, 0.0, 1.0)

    clustering = SpectralClustering(
        n_clusters=num_clients,
        affinity="precomputed",
        random_state=42,
        assign_labels="kmeans",
    )
    labels = clustering.fit_predict(affinity)

    client_clusters: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for node_idx, cluster_id in enumerate(labels):
        client_clusters[int(cluster_id)].append(int(node_idx))

    return client_clusters

def generate_spatial_skew_scenario(
    train_x: np.ndarray,
    train_y: np.ndarray,
    client_clusters: Dict[int, List[int]],
    output_path: str = "splits/spatial_skew.json"
) -> Dict[str, Any]:
    """Kịch bản Non-IID 1: Spatial Skew / Feature Skew.

    TODO:
    - Tính toán lưu lượng trung bình của từng cụm sensor.
    - Sắp xếp và phân bổ để tạo sự chênh lệch phân phối rõ rệt giữa các Client
      (ví dụ: Client 0 toàn cụm nội đô đông đúc, Client 4 toàn cụm ngoại ô thưa xe).
    - Lưu cấu trúc index cố định ra file JSON.
    """
    # TODO: Cài đặt logic kịch bản Spatial Skew tại đây
    num_clients = len(client_clusters)
    cluster_means = {}

    for c_id, node_list in client_clusters.items():
        if len(node_list) > 0:
            # Tính lưu lượng trung bình của cụm node
            avg_flow = np.mean(train_y[:, :, node_list, 0])
            cluster_means[c_id] = float(avg_flow)
        else:
            cluster_means[c_id] = 0.0

    # Sắp xếp cluster theo lưu lượng tăng dần
    sorted_clusters = sorted(cluster_means.keys(), key=lambda k: cluster_means[k])

    partition_info: Dict[str, Any] = {
        "scenario": "spatial_skew",
        "num_clients": num_clients,
        "clients": {},
    }

    total_samples = len(train_x)
    for client_idx, cluster_key in enumerate(sorted_clusters):
        partition_info["clients"][str(client_idx)] = {
            "sensor_nodes": client_clusters[cluster_key],
            "average_flow": cluster_means[cluster_key],
            # Toàn bộ dải thời gian nhưng chỉ trên tập sensor cục bộ
            "sample_indices": list(range(total_samples)),
        }

    save_split_indices(partition_info, output_path)
    return partition_info


def generate_temporal_shift_scenario(
    train_x: np.ndarray,
    num_clients: int = 5,
    output_path: str = "splits/temporal_shift.json"
) -> Dict[str, Any]:
    """Kịch bản Non-IID 2: Temporal Shift / Concept Drift.

    TODO:
    - Phân bổ chỉ số mẫu (sample indices) theo khung giờ khác nhau giữa các Client
      (ví dụ: Client giữ dữ liệu giờ cao điểm, Client giữ dữ liệu đêm/thấp điểm).
    - Lưu cấu trúc index cố định ra file JSON.
    """
    # TODO: Cài đặt logic kịch bản Temporal Shift tại đây
    total_samples = len(train_x)
    num_nodes = train_x.shape[2]
    all_sensors = list(range(num_nodes))

    # Chia chuỗi thời gian thành các khối mẫu liên tiếp lệch pha
    block_size = total_samples // num_clients

    partition_info: Dict[str, Any] = {
        "scenario": "temporal_shift",
        "num_clients": num_clients,
        "clients": {},
    }

    for client_idx in range(num_clients):
        start_idx = client_idx * block_size
        end_idx = total_samples if client_idx == num_clients - 1 else (client_idx + 1) * block_size
        indices = list(range(start_idx, end_idx))

        partition_info["clients"][str(client_idx)] = {
            "sensor_nodes": all_sensors,
            "sample_indices": indices,
            "num_samples": len(indices),
        }

    save_split_indices(partition_info, output_path)
    return partition_info


def save_split_indices(data: Dict[str, Any], filepath: str) -> None:
    """Lưu tệp index cấu hình cố định dạng JSON để đảm bảo tính tái lập."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"-> Đã lưu cấu hình phân chia tại: {filepath}")