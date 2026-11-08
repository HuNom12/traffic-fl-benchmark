
import numpy as np
import pandas as pd


class TrafficSplitter:
    """
    Chronological Train / Validation / Test Split.

    Responsibilities:
    1. Convert DataFrame to tensor (T, N, F).
    2. Split data chronologically: 70/10/20.
    3. Calculate mean and std on Train only.
    4. Apply Z-score normalization to all splits.
    5. Generate sliding windows.

    Returns:
        {
            'train_x', 'train_y',
            'val_x', 'val_y',
            'test_x', 'test_y',
            'mean', 'std'
        }
    """

    TRAFFIC_COLUMNS = [
        "total_flow",
        "avg_occupancy",
        "avg_speed"
    ]

    def __init__(
        self,
        data,
        seq_len=12,
        pred_len=12,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2
    ):

        self.data = data

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.mean = None
        self.std = None

        self.timestamps = None
        self.stations = None

        self._validate_config()

    def _validate_config(self):

        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError(
                "seq_len và pred_len phải lớn hơn 0."
            )

        ratios = [
            self.train_ratio,
            self.val_ratio,
            self.test_ratio
        ]

        if any(r <= 0 for r in ratios):
            raise ValueError("Các tỷ lệ phải lớn hơn 0.")

        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(
                "Tổng train_ratio, val_ratio, test_ratio phải bằng 1."
            )

    def _to_tensor(self):
        """
        Convert DataFrame to tensor (T, N, F).

        T: Number of timestamps
        N: Number of stations
        F: Number of traffic features
        """

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(
                "data phải là pandas DataFrame."
            )

        required = (
            ["timestamp", "station"]
            + self.TRAFFIC_COLUMNS
        )

        missing = [
            col for col in required
            if col not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        df = self.data.copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        if df["timestamp"].isna().any():
            raise ValueError("Có timestamp không hợp lệ.")

        if df.duplicated(
            subset=["timestamp", "station"]
        ).any():
            raise ValueError(
                "Có bản ghi trùng timestamp-station."
            )

        df = df.sort_values(
            ["timestamp", "station"]
        )

        self.timestamps = np.sort(
            df["timestamp"].unique()
        )

        self.stations = np.sort(
            df["station"].unique()
        )

        # Pivot: columns có thứ tự (feature, station)
        wide = df.pivot(
            index="timestamp",
            columns="station",
            values=self.TRAFFIC_COLUMNS
        )

        # Đảm bảo thứ tự timestamp và station nhất quán
        expected_columns = pd.MultiIndex.from_product(
            [self.TRAFFIC_COLUMNS, self.stations]
        )

        wide = wide.reindex(
            index=self.timestamps,
            columns=expected_columns
        )

        # Shape ban đầu: (T, F, N)
        array = wide.to_numpy(
            dtype=np.float32
        ).reshape(
            len(self.timestamps),
            len(self.TRAFFIC_COLUMNS),
            len(self.stations)
        )

        # Chuyển thành (T, N, F)
        array = array.transpose(0, 2, 1)

        if not np.isfinite(array).all():
            raise ValueError(
                "Dữ liệu vẫn còn NaN hoặc Inf. "
                "Hãy hoàn thành imputation trong transform.py "
                "trước khi thực hiện split."
            )

        return array

    def _normalize(self, train, val, test):
        """
        Calculate mean/std on Train only.
        Normalize Train, Validation and Test.
        """

        # Tính trên trục thời gian và station
        # Giữ riêng mean/std cho từng feature
        self.mean = train.mean(
            axis=(0, 1),
            keepdims=True
        )

        self.std = train.std(
            axis=(0, 1),
            keepdims=True
        )

        # Tránh chia cho 0 nếu feature không biến động
        self.std = np.where(
            self.std == 0,
            1.0,
            self.std
        )

        train = (train - self.mean) / self.std
        val = (val - self.mean) / self.std
        test = (test - self.mean) / self.std

        return train, val, test

    def _create_windows(self, data):
        """
        Generate sliding windows.

        X: (samples, seq_len, N, F)
        Y: (samples, pred_len, N, F)
        """

        X = []
        Y = []

        total_length = (
            len(data) - self.seq_len - self.pred_len + 1
        )

        for i in range(max(0, total_length)):

            x = data[
                i : i + self.seq_len
            ]

            y = data[
                i + self.seq_len :
                i + self.seq_len + self.pred_len
            ]

            X.append(x)
            Y.append(y)

        if len(X) == 0:
            return (
                np.empty(
                    (0, self.seq_len, data.shape[1], data.shape[2]),
                    dtype=np.float32
                ),
                np.empty(
                    (0, self.pred_len, data.shape[1], data.shape[2]),
                    dtype=np.float32
                )
            )

        return (
            np.asarray(X, dtype=np.float32),
            np.asarray(Y, dtype=np.float32)
        )

    def run(self):
        """
        Execute chronological split and normalization.

        Returns:
            Dict:
                train_x, train_y
                val_x, val_y
                test_x, test_y
                mean, std
        """

        # 1. Convert to tensor
        data = self._to_tensor()

        total_time = len(data)

        if total_time < self.seq_len + self.pred_len:
            raise ValueError(
                "Không đủ timestamp để tạo sliding windows."
            )

        # 2. Chronological split
        train_end = int(
            total_time * self.train_ratio
        )

        val_end = int(
            total_time * (
                self.train_ratio + self.val_ratio
            )
        )

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        # Kiểm tra độ dài từng tập
        for name, split_data in [
            ("Train", train_data),
            ("Validation", val_data),
            ("Test", test_data)
        ]:
            if len(split_data) < self.seq_len + self.pred_len:
                raise ValueError(
                    f"{name} không đủ dữ liệu để tạo windows."
                )

        print("\nChronological Split")
        print(f"Total timestamps: {total_time}")
        print(f"Train: {len(train_data)}")
        print(f"Validation: {len(val_data)}")
        print(f"Test: {len(test_data)}")

        # 3. Normalize using Train statistics only
        train_data, val_data, test_data = (
            self._normalize(
                train_data,
                val_data,
                test_data
            )
        )

        # 4. Generate sliding windows separately
        train_x, train_y = self._create_windows(
            train_data
        )

        val_x, val_y = self._create_windows(
            val_data
        )

        test_x, test_y = self._create_windows(
            test_data
        )

        # 5. Report shapes
        print("\nSliding Window Shapes")
        print("train_x:", train_x.shape)
        print("train_y:", train_y.shape)
        print("val_x:", val_x.shape)
        print("val_y:", val_y.shape)
        print("test_x:", test_x.shape)
        print("test_y:", test_y.shape)

        print("\nNormalization Statistics")
        for i, feature in enumerate(self.TRAFFIC_COLUMNS):
            print(
                f"{feature}: "
                f"mean={self.mean[0, 0, i]:.6f}, "
                f"std={self.std[0, 0, i]:.6f}"
            )

        return {
            "train_x": train_x,
            "train_y": train_y,

            "val_x": val_x,
            "val_y": val_y,

            "test_x": test_x,
            "test_y": test_y,

            "mean": self.mean.reshape(-1),
            "std": self.std.reshape(-1)
        }
