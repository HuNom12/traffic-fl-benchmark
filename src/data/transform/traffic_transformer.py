
from pathlib import Path
import re

import numpy as np
import pandas as pd


class TrafficTransformer:
    """
    Chuyển dữ liệu METR-LA từ DataFrame thành Tensor 4D.

    X: (B, 12, 207, 2)
    Y: (B, 12, 207, 1)
    """

    def __init__(
        self,
        num_nodes: int = 207,
        input_steps: int = 12,
        output_steps: int = 12,
    ):
        self.num_nodes = num_nodes
        self.input_steps = input_steps
        self.output_steps = output_steps

        self.X = None
        self.Y = None
        self.timestamps = None

    @staticmethod
    def _get_time_columns(df, prefix, feature):
        """
        Lấy và sắp xếp cột theo thứ tự thời gian.

        Ví dụ:
            x_t-11_d0, ..., x_t+0_d0
            y_t+1_d0, ..., y_t+12_d0
        """

        pattern = re.compile(
            rf"^{prefix}_t([+-]?\d+)_d{feature}$"
        )

        columns = []

        for col in df.columns:
            match = pattern.match(col)

            if match:
                offset = int(match.group(1))
                columns.append((offset, col))

        columns.sort(key=lambda item: item[0])

        return columns

    def _extract_columns(self, df):
        """Xác định các cột X và Y."""

        x_d0 = self._get_time_columns(df, "x", 0)
        x_d1 = self._get_time_columns(df, "x", 1)
        y_d0 = self._get_time_columns(df, "y", 0)

        expected_x_offsets = list(
            range(-self.input_steps + 1, 1)
        )

        expected_y_offsets = list(
            range(1, self.output_steps + 1)
        )

        if [x[0] for x in x_d0] != expected_x_offsets:
            raise ValueError(
                f"X d0 offsets không hợp lệ: {x_d0}"
            )

        if [x[0] for x in x_d1] != expected_x_offsets:
            raise ValueError(
                f"X d1 offsets không hợp lệ: {x_d1}"
            )

        if [x[0] for x in y_d0] != expected_y_offsets:
            raise ValueError(
                f"Y d0 offsets không hợp lệ: {y_d0}"
            )

        return (
            [x[1] for x in x_d0],
            [x[1] for x in x_d1],
            [x[1] for x in y_d0],
        )

    def validate_structure(self, df):
        """Kiểm tra cấu trúc dữ liệu trước khi reshape."""

        required = {"t0_timestamp", "node_id"}

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Thiếu các cột bắt buộc: {missing}"
            )

        if df["node_id"].isna().any():
            raise ValueError("node_id chứa giá trị null.")

        if df["t0_timestamp"].isna().any():
            raise ValueError("t0_timestamp chứa giá trị null.")

        if df.duplicated(
            subset=["t0_timestamp", "node_id"]
        ).any():
            raise ValueError(
                "Có node_id bị trùng trong cùng timestamp."
            )

        node_ids = set(df["node_id"].unique())

        expected_nodes = set(range(self.num_nodes))

        if node_ids != expected_nodes:
            raise ValueError(
                "Node IDs không khớp với tập "
                f"0..{self.num_nodes - 1}."
            )

        counts = df.groupby(
            "t0_timestamp",
            sort=False
        )["node_id"].count()

        invalid = counts[counts != self.num_nodes]

        if not invalid.empty:
            raise ValueError(
                f"Có {len(invalid)} timestamp không đủ "
                f"{self.num_nodes} node."
            )

        return self

    def transform(self, df):
        """
        Chuyển DataFrame thành X và Y.

        Thứ tự:
            timestamp tăng dần
            node_id tăng dần
            thời gian tăng dần
        """

        self.validate_structure(df)

        x_d0, x_d1, y_d0 = self._extract_columns(df)

        df = df.copy()

        # Sắp xếp theo thời gian và node.
        df = df.sort_values(
            ["t0_timestamp", "node_id"]
        ).reset_index(drop=True)

        # Đảm bảo node_id của từng sample là 0..206.
        node_matrix = df["node_id"].to_numpy().reshape(
            -1, self.num_nodes
        )

        expected = np.arange(self.num_nodes)

        if not np.all(node_matrix == expected):
            raise ValueError(
                "Thứ tự node sau khi sort không hợp lệ."
            )

        timestamps = (
            df["t0_timestamp"]
            .drop_duplicates()
            .to_numpy()
        )

        num_samples = len(timestamps)

        # X: (B * N, 12, 2)
        x_d0_values = df[x_d0].to_numpy(
            dtype=np.float32
        )

        x_d1_values = df[x_d1].to_numpy(
            dtype=np.float32
        )

        x_values = np.stack(
            [x_d0_values, x_d1_values],
            axis=-1
        )

        # (B, N, 12, 2) -> (B, 12, N, 2)
        X = x_values.reshape(
            num_samples,
            self.num_nodes,
            self.input_steps,
            2
        ).transpose(0, 2, 1, 3)

        # Y: chỉ lấy d0.
        y_values = df[y_d0].to_numpy(
            dtype=np.float32
        )

        # (B, N, 12) -> (B, 12, N, 1)
        Y = y_values.reshape(
            num_samples,
            self.num_nodes,
            self.output_steps
        ).transpose(0, 2, 1)

        Y = Y[..., np.newaxis]

        if not np.isfinite(X).all():
            raise ValueError("X chứa NaN hoặc Inf.")

        if not np.isfinite(Y).all():
            raise ValueError("Y chứa NaN hoặc Inf.")

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.timestamps = timestamps

        print("Traffic transformation completed.")
        print("X shape:", self.X.shape)
        print("Y shape:", self.Y.shape)

        return self.X, self.Y

    def get_data(self):
        if self.X is None or self.Y is None:
            raise ValueError("Chưa thực hiện transform.")

        return self.X, self.Y
