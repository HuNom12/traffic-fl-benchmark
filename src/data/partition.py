import os
import json
import numpy as np
from typing import Dict, List, Any


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
    raise NotImplementedError("Cần hoàn thiện hàm partition_nodes_spatial_clustering")


def generate_spatial_skew_scenario(
    traffic_data: np.ndarray,
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
    raise NotImplementedError("Cần hoàn thiện hàm generate_spatial_skew_scenario")


def generate_temporal_shift_scenario(
    num_samples: int,
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
    raise NotImplementedError("Cần hoàn thiện hàm generate_temporal_shift_scenario")


def save_split_indices(data: Dict[str, Any], filepath: str) -> None:
    """Lưu tệp index cấu hình cố định dạng JSON để đảm bảo tính tái lập."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"-> Đã lưu cấu hình phân chia tại: {filepath}")