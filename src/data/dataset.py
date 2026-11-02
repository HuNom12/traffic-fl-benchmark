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
    raise NotImplementedError("Cần hoàn thiện hàm generate_sliding_window")


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
    # TODO: Cài đặt logic tại đây
    raise NotImplementedError("Cần hoàn thiện hàm load_and_preprocess_pems")