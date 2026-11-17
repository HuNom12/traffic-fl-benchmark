
from pathlib import Path

import numpy as np
import pandas as pd

from .traffic_transformer import TrafficTransformer
from .feature_transformer import FeatureTransformer


class DatasetTransformer:

    def __init__(
        self,
        raw_dir,
        output_dir,
        num_nodes=207,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

        self.num_nodes = num_nodes

        self.traffic_transformer = TrafficTransformer(
            num_nodes=num_nodes,
            input_steps=12,
            output_steps=12,
        )

        self.feature_transformer = FeatureTransformer()

        self.datasets = {}

    def _get_split_path(self, split):
        """
        Hỗ trợ cả val.parquet và validation.parquet.
        """

        candidates = {
            "train": ["train.parquet"],
            "val": ["val.parquet", "validation.parquet"],
            "test": ["test.parquet"],
        }

        for filename in candidates[split]:
            path = self.raw_dir / filename

            if path.exists():
                return path

        raise FileNotFoundError(
            f"Không tìm thấy file cho split: {split}"
        )

    def load_split(self, split):
        path = self._get_split_path(split)

        print(f"\nLoading {split}: {path}")

        return pd.read_parquet(path)

    def transform_splits(self):
        for split in ["train", "val", "test"]:
            df = self.load_split(split)

            X, Y = self.traffic_transformer.transform(df)

            self.datasets[split] = {
                "X": X,
                "Y": Y,
            }

            print(
                f"{split}: "
                f"X={X.shape}, Y={Y.shape}"
            )

        return self

    def normalize(self):
        """
        Fit scaler trên X_train[..., 0].
        Dùng cùng scaler cho X và Y của mọi split.
        """

        X_train = self.datasets["train"]["X"]

        self.feature_transformer.fit(X_train)

        for split in ["train", "val", "test"]:
            X = self.datasets[split]["X"]
            Y = self.datasets[split]["Y"]

            self.datasets[split]["X"] = (
                self.feature_transformer.transform_x(X)
            )

            self.datasets[split]["Y"] = (
                self.feature_transformer.transform_y(Y)
            )

        return self

    def save(self):
        for split in ["train", "val", "test"]:
            split_dir = self.output_dir / split
            split_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            X = self.datasets[split]["X"]
            Y = self.datasets[split]["Y"]

            np.save(
                split_dir / f"X_{split}.npy",
                X
            )

            np.save(
                split_dir / f"Y_{split}.npy",
                Y
            )

        self.feature_transformer.save(
            self.output_dir / "scaler.pkl"
        )

        print("\nAll datasets saved.")

        return self

    def validate_output(self):
        expected_shapes = {
            "X": (12, self.num_nodes, 2),
            "Y": (12, self.num_nodes, 1),
        }

        for split, data in self.datasets.items():
            X = data["X"]
            Y = data["Y"]

            if X.ndim != 4 or Y.ndim != 4:
                raise ValueError(
                    f"{split}: Tensor không có 4 chiều."
                )

            if X.shape[1:] != expected_shapes["X"]:
                raise ValueError(
                    f"{split}: X shape sai: {X.shape}"
                )

            if Y.shape[1:] != expected_shapes["Y"]:
                raise ValueError(
                    f"{split}: Y shape sai: {Y.shape}"
                )

            if not np.isfinite(X).all():
                raise ValueError(f"{split}: X không hữu hạn.")

            if not np.isfinite(Y).all():
                raise ValueError(f"{split}: Y không hữu hạn.")

            print(
                f"{split} validated: "
                f"X={X.shape}, Y={Y.shape}"
            )

        return self

    def run(self):
        (
            self
            .transform_splits()
            .normalize()
            .validate_output()
            .save()
        )

        return self
