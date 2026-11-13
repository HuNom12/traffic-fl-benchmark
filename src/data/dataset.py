
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class StandardTrafficDataset(Dataset):
    """
    PyTorch Dataset cho dữ liệu METR-LA.

    X: (B, 12, 207, 2)
    Y: (B, 12, 207, 1)
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("X và Y phải là tensor 4 chiều")

        if len(x) != len(y):
            raise ValueError("Số lượng samples của X và Y không khớp")

        if x.shape[2] != 207 or y.shape[2] != 207:
            raise ValueError("METR-LA phải có 207 nodes")

        if x.shape[1:] != (12, 207, 2):
            raise ValueError(f"Shape X không hợp lệ: {x.shape}")

        if y.shape[1:] != (12, 207, 1):
            raise ValueError(f"Shape Y không hợp lệ: {y.shape}")

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("X hoặc Y chứa NaN/Inf")

        self.x = torch.from_numpy(
            np.asarray(x, dtype=np.float32)
        )

        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.x[idx], self.y[idx]


def load_metr_la(
    processed_dir: str = "data/processed/METR-LA",
) -> Dict[str, Any]:
    """
    Load dữ liệu METR-LA đã được xử lý bởi Transform Pipeline.

    Không thực hiện:
    - Chia lại Train/Val/Test.
    - Tạo sliding window.
    - Chuẩn hóa lại dữ liệu.
    """

    root = Path(processed_dir)

    files = {
        "train_x": root / "train/X_train.npy",
        "train_y": root / "train/Y_train.npy",
        "val_x": root / "val/X_val.npy",
        "val_y": root / "val/Y_val.npy",
        "test_x": root / "test/X_test.npy",
        "test_y": root / "test/Y_test.npy",
        "adjacency": root / "graph/adjacency_matrix.npy",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {name}: {path}"
            )

    data = {
        name: np.load(path)
        for name, path in files.items()
    }

    expected_shapes = {
        "train_x": (12, 207, 2),
        "train_y": (12, 207, 1),
        "val_x": (12, 207, 2),
        "val_y": (12, 207, 1),
        "test_x": (12, 207, 2),
        "test_y": (12, 207, 1),
    }

    for name, expected in expected_shapes.items():
        if data[name].shape[1:] != expected:
            raise ValueError(
                f"{name} có shape {data[name].shape}, "
                f"mong đợi (B, {expected})"
            )

    adjacency = data["adjacency"]

    if adjacency.shape != (207, 207):
        raise ValueError(
            f"Adjacency matrix không hợp lệ: {adjacency.shape}"
        )

    for name, array in data.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} chứa NaN hoặc Inf")

    return data


def create_dataloaders(
    data: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Tạo DataLoader cho Train, Validation và Test.
    """

    datasets = {
        "train": StandardTrafficDataset(
            data["train_x"],
            data["train_y"],
        ),
        "val": StandardTrafficDataset(
            data["val_x"],
            data["val_y"],
        ),
        "test": StandardTrafficDataset(
            data["test_x"],
            data["test_y"],
        ),
    }

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
