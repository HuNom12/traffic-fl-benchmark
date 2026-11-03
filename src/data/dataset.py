import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any


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

    TODO :
    1. Đọc khoảng cách giữa các cặp sensor từ file CSV.
    2. Chuẩn hóa khoảng cách hoặc chia cho độ lệch chuẩn.
    3. Áp dụng công thức: A_ij = exp(- (dist(vi, vj) / sigma)^2) nếu dist <= kappa, ngược lại 0.
    4. Đảm bảo đường chéo chính bằng 1 (hoặc theo quy ước đồ thị).
    
    Returns:
        Ma trận kề A kích thước (num_nodes, num_nodes)
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if not os.path.exists(distance_file_path):
        # Fallback: sinh ma trận đơn vị + kết nối lân cận nếu chưa có file thực tế
        adj = np.eye(num_nodes, dtype=np.float32)
        for i in range(num_nodes - 1):
            adj[i, i + 1] = adj[i + 1, i] = 0.5
        return adj
    df = pd.read_csv(distance_file_path)
    col_from, col_to, col_cost = df.columns[0], df.columns[1], df.columns[2]
    # Lấy khoảng cách hợp lệ để tính độ lệch chuẩn khoảng cách
    distances = df[col_cost].values
    distances = distances[~np.isnan(distances)]
    dist_std = np.std(distances) if len(distances) > 0 else 1.0

    for _, row in df.iterrows():
        u, v, d = int(row[col_from]), int(row[col_to]), float(row[col_cost])
        if u < num_nodes and v < num_nodes:
            # Chuẩn hóa khoảng cách theo std
            norm_dist = d / (dist_std + 1e-5)
            weight = np.exp(- (norm_dist / sigma) ** 2)
            if weight >= kappa:
                adj[u, v] = weight
                adj[v, u] = weight

    # Đường chéo chính tự kết nối (self-loop)
    for i in range(num_nodes):
        adj[i, i] = 1.0
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
    total_timesteps, num_nodes, num_features = data.shape
    num_samples = total_timesteps - time_in - time_out + 1

    x = np.zeros((num_samples, time_in, num_nodes, num_features), dtype=np.float32)
    y = np.zeros((num_samples, time_out, num_nodes, 1), dtype=np.float32)

    for i in range(num_samples):
        x[i] = data[i : i + time_in]
        # Y chỉ lấy kênh 0 (Traffic Flow)
        y[i] = data[i + time_in : i + time_in + time_out, :, :1]

    return x, y

def load_and_preprocess_pems(
    raw_data_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    time_in: int = 12,
    time_out: int = 12
) -> Dict[str, Any]:
    """Quy trình tiền xử lý dữ liệu PeMS.

    TODO:
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
    # TODO: Cài đặt logic tại đây
    if os.path.exists(raw_data_path):
        raw_npz = np.load(raw_data_path)
        # File PeMS npz thường lưu dưới key 'data'
        data = raw_npz["data"].astype(np.float32)
    else:
        # Giả lập dữ liệu PeMS04 (307 sensors, 3 features: flow, occupy, speed) để test pipeline
        total_steps = 1000
        nodes = 307
        data = np.random.uniform(20.0, 150.0, size=(total_steps, nodes, 3)).astype(np.float32)

    # 1. Xử lý giá trị 0 hoặc NaN bằng nội suy tuyến tính theo trục thời gian
    for node in range(data.shape[1]):
        for feat in range(data.shape[2]):
            series = pd.Series(data[:, node, feat])
            series = series.replace(0.0, np.nan).interpolate(method="linear").bfill().ffill()
            data[:, node, feat] = series.values

    # 2. Phân chia chuỗi thời gian tuyến tính (không shuffle chống rò rỉ)
    total_timesteps = len(data)
    train_end = int(total_timesteps * train_ratio)
    val_end = int(total_timesteps * (train_ratio + val_ratio))

    train_raw = data[:train_end]
    val_raw = data[train_end:val_end]
    test_raw = data[val_end:]

    # 3. Tính Mean và Std CHỈ TRÊN TẬP TRAIN (chuẩn hóa kênh 0 - Flow)
    mean = np.mean(train_raw[:, :, 0])
    std = np.std(train_raw[:, :, 0])
    if std == 0.0:
        std = 1.0

    # Chuẩn hóa Z-score toàn bộ dữ liệu theo thống kê tập Train
    train_norm = np.copy(train_raw)
    val_norm = np.copy(val_raw)
    test_norm = np.copy(test_raw)

    train_norm[:, :, 0] = (train_raw[:, :, 0] - mean) / std
    val_norm[:, :, 0] = (val_raw[:, :, 0] - mean) / std
    test_norm[:, :, 0] = (test_raw[:, :, 0] - mean) / std

    # 4. Cắt lát Sliding Window
    train_x, train_y = generate_sliding_window(train_norm, time_in, time_out)
    val_x, val_y = generate_sliding_window(val_norm, time_in, time_out)
    test_x, test_y = generate_sliding_window(test_norm, time_in, time_out)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "mean": float(mean),
        "std": float(std),
    }