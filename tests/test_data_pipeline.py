import os
import pickle
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_metr_la, StandardTrafficDataset


PROCESSED_DIR = os.path.join("data", "processed", "METR-LA")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED_DIR, "train", "X_train.npy")),
    reason="Dữ liệu METR-LA đã xử lý chưa tồn tại. Hãy chạy `python -m src.data.data_pipeline` trước."
)
def test_metr_la_processed_artifacts_integrity():
    """Kiểm tra tính toàn vẹn của các tệp đầu ra sau TransformPipeline qua load_metr_la()."""
    data = load_metr_la(PROCESSED_DIR)

    X_train, Y_train = data["train_x"], data["train_y"]
    X_val, Y_val = data["val_x"], data["val_y"]
    X_test, Y_test = data["test_x"], data["test_y"]
    adj = data["adjacency"]

    # 1. Kiểm tra kích thước (Shape) chuẩn của METR-LA: (B, 12, 207, C)
    assert X_train.ndim == 4 and X_train.shape[1:] == (12, 207, 2)
    assert Y_train.ndim == 4 and Y_train.shape[1:] == (12, 207, 1)
    assert X_train.shape[0] == Y_train.shape[0]

    assert X_val.shape[1:] == (12, 207, 2)
    assert Y_val.shape[1:] == (12, 207, 1)

    assert X_test.shape[1:] == (12, 207, 2)
    assert Y_test.shape[1:] == (12, 207, 1)

    # 2. Kiểm tra ma trận kề A (207, 207)
    assert adj.shape == (207, 207)
    assert np.isfinite(adj).all()
    assert adj.min() >= 0.0 and adj.max() <= 1.0

    # 3. Kiểm tra Scaler đã lưu từ tập Train
    scaler_path = os.path.join(PROCESSED_DIR, "scaler.pkl")
    assert os.path.exists(scaler_path), "Thiếu tệp scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    assert scaler is not None


def test_standard_traffic_dataset_and_dataloader():
    """Kiểm tra StandardTrafficDataset và DataLoader với dữ liệu chuẩn hóa METR-LA."""
    B, T_in, T_out, N = 32, 12, 12, 207
    X_mock = np.random.randn(B, T_in, N, 2).astype(np.float32)
    Y_mock = np.random.randn(B, T_out, N, 1).astype(np.float32)

    dataset = StandardTrafficDataset(X_mock, Y_mock)
    assert len(dataset) == B

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    batch_x, batch_y = next(iter(loader))

    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)
    assert batch_x.shape == (8, 12, 207, 2)
    assert batch_y.shape == (8, 12, 207, 1)