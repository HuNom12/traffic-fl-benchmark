
import pickle
from pathlib import Path

import numpy as np


class FeatureTransformer:
    """
    Chuẩn hóa Z-score trên feature d0.

    Mean và Std chỉ được tính từ X_train[..., 0].

    d1 được giữ nguyên.
    """

    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, X_train):
        """
        Tính Mean và Std từ tốc độ trong X_train.
        """

        speed = X_train[..., 0]

        self.mean = float(np.mean(speed))
        self.std = float(np.std(speed))

        if self.std < self.eps:
            raise ValueError(
                "Standard deviation quá nhỏ."
            )

        print("Scaler fitted.")
        print("Mean:", self.mean)
        print("Std:", self.std)

        return self

    def transform_x(self, X):
        """
        Chuẩn hóa d0 của X.
        Giữ nguyên d1.
        """

        if self.mean is None or self.std is None:
            raise ValueError("Scaler chưa được fit.")

        result = X.copy()

        result[..., 0] = (
            result[..., 0] - self.mean
        ) / self.std

        return result.astype(np.float32)

    def transform_y(self, Y):
        """
        Chuẩn hóa target speed bằng scaler của Train.
        """

        if self.mean is None or self.std is None:
            raise ValueError("Scaler chưa được fit.")

        result = (
            Y - self.mean
        ) / self.std

        return result.astype(np.float32)

    def save(self, path):
        """Lưu scaler."""

        if self.mean is None or self.std is None:
            raise ValueError("Scaler chưa được fit.")

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "feature": "d0",
                    "method": "z-score",
                },
                f
            )

        print(f"Scaler saved: {path}")

    def load(self, path):
        """Load scaler đã lưu."""

        with open(path, "rb") as f:
            params = pickle.load(f)

        self.mean = params["mean"]
        self.std = params["std"]

        return self
