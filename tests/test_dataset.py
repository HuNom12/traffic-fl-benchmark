import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_metr_la, StandardTrafficDataset


PROCESSED_DIR = os.path.join("data", "processed", "METR-LA")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED_DIR, "train", "X_train.npy")),
    reason="Dữ liệu METR-LA đã xử lý chưa tồn tại."
)
def test_dataset_values_and_feature_ranges():
    """Kiểm tra giá trị hữu hạn và miền giá trị của Speed (d0) và Time-of-Day (d1)."""
    data = load_metr_la(PROCESSED_DIR)
    X_train, Y_train = data["train_x"], data["train_y"]
    X_val, Y_val = data["val_x"], data["val_y"]
    X_test, Y_test = data["test_x"], data["test_y"]

    # 1. Không chứa NaN hoặc Inf
    for name, arr in [
        ("X_train", X_train), ("Y_train", Y_train),
        ("X_val", X_val), ("Y_val", Y_val),
        ("X_test", X_test), ("Y_test", Y_test),
    ]:
        assert np.isfinite(arr).all(), f"{name} chứa giá trị NaN hoặc Inf"

    # 2. Kênh 0 (Speed) đã được chuẩn hóa Z-score trên tập Train (mean ~ 0, std ~ 1)
    speed_train = X_train[..., 0]
    assert abs(float(speed_train.mean())) < 0.05, f"Mean của speed chuẩn hóa bất thường: {speed_train.mean()}"
    assert abs(float(speed_train.std()) - 1.0) < 0.05, f"Std của speed chuẩn hóa bất thường: {speed_train.std()}"

    # 3. Kênh 1 (Time-of-Day) phải nằm trong đoạn [0.0, 1.0]
    tod_train = X_train[..., 1]
    assert tod_train.min() >= -1e-5 and tod_train.max() <= 1.00001, (
        f"Time-of-day nằm ngoài [0, 1]: min={tod_train.min()}, max={tod_train.max()}"
    )


@pytest.mark.parametrize("batch_size", [16, 32])
def test_dataloader_batching_with_subgraphs(batch_size: int):
    """Kiểm tra StandardTrafficDataset và DataLoader với cả đồ thị toàn cục (207) và đồ thị con (10)."""
    X_mock = np.random.randn(64, 12, 207, 2).astype(np.float32)
    Y_mock = np.random.randn(64, 12, 207, 1).astype(np.float32)

    dataset = StandardTrafficDataset(X_mock, Y_mock)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bx, by = next(iter(loader))
    assert bx.shape == (batch_size, 12, 207, 2)
    assert by.shape == (batch_size, 12, 207, 1)
    assert bx.dtype == torch.float32
    assert by.dtype == torch.float32