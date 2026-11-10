import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from typing import Tuple, Dict, Any
from src.data.loader import TrafficLoader
from src.data.transform import TrafficTransformer

class StandardTrafficDataset(Dataset):
    """PyTorch Dataset cho dữ liệu giao thông chuỗi thời gian dạng trượt (Sliding Window)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: Tensor mảng đặc trưng đầu vào (Num_samples, Time_in, Num_nodes, Features)
            y: Tensor mảng mục tiêu dự báo (Num_samples, Time_out, Num_nodes, 1)
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def compute_adjacency_matrix(
    distance_file_path: str,
    num_nodes: int,
    sigma: float = 0.1,
    kappa: float = 0.5
) -> np.ndarray:
    """Tính ma trận kề A theo hàm ngưỡng Gaussian (Thresholded Gaussian Kernel).

    TODO:
    1. Đọc khoảng cách giữa các cặp sensor từ file CSV.
    2. Chuẩn hóa khoảng cách hoặc chia cho độ lệch chuẩn.
    3. Áp dụng công thức: A_ij = exp(- (dist(vi, vj) / sigma)^2) nếu dist <= kappa, ngược lại 0.
    4. Đảm bảo đường chéo chính bằng 1 (hoặc theo quy ước đồ thị).
    
    Returns:
        Ma trận kề A kích thước (num_nodes, num_nodes)
    """
    adj = np.eye(num_nodes, dtype=np.float32)
    # TODO: Cài đặt logic đọc distance và tính kernel tại đây
    return adj


def generate_sliding_window(
    data: np.ndarray,
    time_in: int = 12,
    time_out: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """Cắt lát chuỗi thời gian thành các cặp (X, Y) theo cửa sổ trượt.

    TODO (Task 1.2):
    - Input data gốc shape: (Total_timesteps, Num_nodes, Features)
    - X có kích thước: (Num_samples, time_in, Num_nodes, Features)
    - Y có kích thước: (Num_samples, time_out, Num_nodes, 1) (chỉ lấy kênh lưu lượng)
    """
    # TODO: Cài đặt logic cắt lát sliding window tại đây
        # 1. Kiểm tra dữ liệu đầu vào
    if not isinstance(data, np.ndarray):
        raise TypeError("data phải là numpy.ndarray")

    if data.ndim != 3:
        raise ValueError("data phải có shape (T, N, F)")

    if time_in <= 0 or time_out <= 0:
        raise ValueError("time_in và time_out phải lớn hơn 0")

    T, N, F = data.shape

    if F < 1:
        raise ValueError("Dữ liệu phải có ít nhất một feature")

    if T < time_in + time_out:
        raise ValueError("Không đủ timestep để tạo sliding window")

    X, Y = [], []

    num_samples = T - time_in - time_out + 1

    for i in range(num_samples):
        x = data[i:i + time_in]

        # Channel 0 là total_flow
        y = data[
            i + time_in:i + time_in + time_out,
            :,
            0:1
        ]

        X.append(x)
        Y.append(y)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32)
    )


def load_and_preprocess_pems(
    raw_data_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    time_in: int = 12,
    time_out: int = 12
) -> Dict[str, Any]:
    """Quy trình tiền xử lý dữ liệu PeMS:

    TODO (Task 1.2):
    1. Đọc file .npz gốc (ví dụ kênh 0 là traffic flow).
    2. Nội suy tuyến tính các giá trị 0 hoặc NaN.
    3. Phân chia chuỗi thời gian tuần tự theo tỷ lệ train/val/test (không shuffle).
    4. Tính mean và std TRÊN TẬP TRAIN, áp dụng Z-score chuẩn hóa cho train, val, test.
    5. Tạo sliding window cho từng tập.

    Returns:
        Dict chứa:
        - 'train_x', 'train_y', 'val_x', 'val_y', 'test_x', 'test_y'
        - 'mean', 'std' (để sau này denormalize khi tính MAE/RMSE)
    """
    # ==========================================
    # 1. VALIDATE CONFIGURATION
    # ==========================================

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Tổng train, val, test ratio phải bằng 1")

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Các tỷ lệ phải lớn hơn 0")

    # ==========================================
    # 2. LOAD DATA
    # ==========================================

    loader = TrafficLoader()

    df = loader.load(raw_data_path)

    print("Raw data shape:", df.shape)

    # ==========================================
    # 3. TRANSFORM DATA
    # ==========================================

    transformer = TrafficTransformer(df)

    transformer = (
        transformer
        .remove_lane_features()
        .sort_data()
        .remove_stations_without_traffic()
        .impute_flow()
        .flow_based_imputation()
    )

    transformed_df = transformer.get_data()

    transformer.save(
        "../../data/processed/transformed/traffic_transformed1.parquet"
    )

    # ==========================================
    # 4. DATAFRAME -> NUMPY TENSOR
    # ==========================================

    features = [
        "total_flow",
        "avg_occupancy",
        "avg_speed"
    ]

    # Kiểm tra dữ liệu
    required_columns = ["timestamp", "station"] + features

    missing_columns = [
        col for col in required_columns
        if col not in transformed_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Thiếu các cột: {missing_columns}")

    # Kiểm tra trùng timestamp-station
    if transformed_df.duplicated(["timestamp", "station"]).any():
        raise ValueError("Dữ liệu có timestamp-station bị trùng")

    # Sắp xếp station và timestamp
    stations = np.sort(transformed_df["station"].unique())
    timestamps = np.sort(transformed_df["timestamp"].unique())

    # Pivot thành dạng rộng
    pivoted = transformed_df.pivot(
        index="timestamp",
        columns="station",
        values=features
    )

    # Đồng bộ thứ tự timestamp và station
    pivoted = pivoted.reindex(
        index=timestamps,
        columns=pd.MultiIndex.from_product(
            [features, stations]
        )
    )

    # Shape ban đầu: (T, F, N)
    # Chuyển thành: (T, N, F)
    data = (
        pivoted.to_numpy(dtype=np.float32)
        .reshape(len(timestamps), len(features), len(stations))
        .transpose(0, 2, 1)
    )

    print("Tensor shape:", data.shape)

    # Không cho phép NaN hoặc Inf đi vào model
    if not np.isfinite(data).all():
        raise ValueError(
            "Tensor vẫn còn NaN hoặc Inf. "
            "Hãy kiểm tra lại bước imputation."
        )

    # ==========================================
    # 5. SEQUENTIAL TRAIN / VAL / TEST SPLIT
    # ==========================================

    total_timesteps = data.shape[0]

    train_end = int(total_timesteps * train_ratio)
    val_end = train_end + int(total_timesteps * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print("Train shape:", train_data.shape)
    print("Val shape:", val_data.shape)
    print("Test shape:", test_data.shape)

    # ==========================================
    # 6. Z-SCORE NORMALIZATION
    # ==========================================

    # Tính mean và std trên Train.
    # Tính riêng cho từng feature, gộp theo thời gian và station.
    mean = train_data.mean(axis=(0, 1), keepdims=True)
    std = train_data.std(axis=(0, 1), keepdims=True)

    # Tránh chia cho 0
    std = np.where(std == 0, 1.0, std)

    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std

    # ==========================================
    # 7. GENERATE SLIDING WINDOWS
    # ==========================================

    train_x, train_y = generate_sliding_window(
        train_data, time_in, time_out
    )

    val_x, val_y = generate_sliding_window(
        val_data, time_in, time_out
    )

    test_x, test_y = generate_sliding_window(
        test_data, time_in, time_out
    )

    print("\n===== SLIDING WINDOW RESULTS =====")
    print("Train X:", train_x.shape)
    print("Train Y:", train_y.shape)
    print("Val X:", val_x.shape)
    print("Val Y:", val_y.shape)
    print("Test X:", test_x.shape)
    print("Test Y:", test_y.shape)

    # ==========================================
    # 8. RETURN RESULTS
    # ==========================================

    return {
        "train_x": train_x,
        "train_y": train_y,

        "val_x": val_x,
        "val_y": val_y,

        "test_x": test_x,
        "test_y": test_y,

        "mean": mean.reshape(-1),
        "std": std.reshape(-1),

        "stations": stations,
        "timestamps": timestamps
    }