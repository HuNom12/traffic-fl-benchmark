
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.dataset import load_metr_la


RAW_DIR = Path("data/raw/METR-LA")
PROCESSED_DIR = Path("data/processed/METR-LA")

NUM_NODES = 207
INPUT_STEPS = 12
OUTPUT_STEPS = 12

ATOL = 1e-5
RTOL = 1e-5


def load_scaler():
    """Load scaler đã được fit trên X_train."""

    scaler_path = PROCESSED_DIR / "scaler.pkl"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    mean = float(np.asarray(scaler["mean"]).reshape(-1)[0])
    std = float(np.asarray(scaler["std"]).reshape(-1)[0])

    assert np.isfinite(mean), "Scaler mean không hợp lệ"
    assert np.isfinite(std) and std > 0, "Scaler std không hợp lệ"

    return mean, std


def load_raw_train():
    """
    Đọc các cột cần thiết từ Parquet gốc.
    Không đọc toàn bộ các cột không liên quan.
    """

    raw_path = RAW_DIR / "train.parquet"

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    parquet = pq.ParquetFile(raw_path)
    all_columns = parquet.schema.names

    x_pattern = re.compile(r"^x_t([+-]\d+)_d([01])$")
    y_pattern = re.compile(r"^y_t([+-]\d+)_d0$")

    x_columns = [
        col for col in all_columns
        if x_pattern.match(col)
    ]

    y_columns = [
        col for col in all_columns
        if y_pattern.match(col)
    ]

    required = ["t0_timestamp", "node_id"] + x_columns + y_columns

    missing = [
        col for col in required
        if col not in all_columns
    ]

    if missing:
        raise ValueError(f"Parquet thiếu cột: {missing}")

    df = pd.read_parquet(
        raw_path,
        columns=required,
    )

    return df


def get_ordered_columns(df, prefix, feature):
    """Lấy tên cột theo thứ tự offset thời gian."""

    pattern = re.compile(
        rf"^{prefix}_t([+-]\d+)_d{feature}$"
    )

    columns = []

    for col in df.columns:
        match = pattern.match(col)

        if match:
            offset = int(match.group(1))
            columns.append((offset, col))

    columns.sort(key=lambda item: item[0])

    return [col for _, col in columns]


def test_train_shapes(data):
    """Kiểm tra cấu trúc X_train và Y_train."""

    X = data["train_x"]
    Y = data["train_y"]

    print("\n========== 1. SHAPE VALIDATION ==========")

    print("X_train:", X.shape)
    print("Y_train:", Y.shape)

    assert X.ndim == 4
    assert Y.ndim == 4

    assert X.shape[1:] == (12, 207, 2)
    assert Y.shape[1:] == (12, 207, 1)

    assert X.shape[0] == Y.shape[0]

    print("PASS: Shape hợp lệ")


def test_feature_columns(df):
    """Kiểm tra đúng các feature và thứ tự timestep."""

    x_d0 = get_ordered_columns(df, "x", 0)
    x_d1 = get_ordered_columns(df, "x", 1)
    y_d0 = get_ordered_columns(df, "y", 0)

    expected_x = list(range(-11, 1))
    expected_y = list(range(1, 13))

    x_d0_offsets = [
        int(re.search(r"_t([+-]\d+)_", c).group(1))
        for c in x_d0
    ]

    x_d1_offsets = [
        int(re.search(r"_t([+-]\d+)_", c).group(1))
        for c in x_d1
    ]

    y_offsets = [
        int(re.search(r"_t([+-]\d+)_", c).group(1))
        for c in y_d0
    ]

    print("\n========== 2. FEATURE VALIDATION ==========")

    print("X speed offsets:", x_d0_offsets)
    print("X time offsets:", x_d1_offsets)
    print("Y speed offsets:", y_offsets)

    assert x_d0_offsets == expected_x
    assert x_d1_offsets == expected_x
    assert y_offsets == expected_y

    print("PASS: Feature và timestep hợp lệ")


def test_node_ordering(df):
    """Kiểm tra mỗi timestamp có đủ 207 nodes."""

    print("\n========== 3. NODE VALIDATION ==========")

    df = df.sort_values(
        ["t0_timestamp", "node_id"]
    )

    node_counts = df.groupby(
        "t0_timestamp",
        sort=False,
    )["node_id"].nunique()

    assert (node_counts == NUM_NODES).all()

    expected_nodes = np.arange(NUM_NODES)

    # Kiểm tra node ordering ở một số timestamp.
    groups = df.groupby("t0_timestamp", sort=False)

    checked = 0

    for _, group in groups:
        nodes = group["node_id"].to_numpy()

        assert np.array_equal(nodes, expected_nodes)

        checked += 1

        if checked == 3:
            break

    print("Number of timestamps:", len(node_counts))
    print("PASS: Mỗi timestamp có đủ 207 nodes")


def build_reference_sample(
    group,
    mean,
    std,
):
    """
    Dựng lại một sample X/Y trực tiếp từ Parquet.

    X: 12 quá khứ x 207 nodes x 2 features
    Y: 12 tương lai x 207 nodes x 1 target
    """

    group = group.sort_values("node_id")

    x_d0_columns = get_ordered_columns(group, "x", 0)
    x_d1_columns = get_ordered_columns(group, "x", 1)
    y_d0_columns = get_ordered_columns(group, "y", 0)

    # Raw X có shape (207, 12).
    speed = group[x_d0_columns].to_numpy(dtype=np.float32)
    time_of_day = group[x_d1_columns].to_numpy(dtype=np.float32)

    # Chuẩn hóa speed bằng scaler của Train.
    speed = (speed - mean) / std

    # Ghép hai features thành (207, 12, 2).
    X = np.stack(
        [speed, time_of_day],
        axis=-1,
    )

    # Chuyển thành (12, 207, 2).
    X = X.transpose(1, 0, 2)

    # Raw Y: (207, 12).
    Y = group[y_d0_columns].to_numpy(dtype=np.float32)

    # Dùng cùng scaler đã fit trên X speed.
    Y = (Y - mean) / std

    # (207, 12) -> (12, 207, 1).
    Y = Y.transpose(1, 0)[..., np.newaxis]

    return X, Y


def test_raw_vs_processed(data, df):
    """
    Đối chiếu dữ liệu đã transform với Parquet gốc.

    Kiểm tra sample đầu, giữa và cuối.
    """

    X_train = data["train_x"]
    Y_train = data["train_y"]

    mean, std = load_scaler()

    print("\n========== 4. RAW VS PROCESSED ==========")
    print("Scaler mean:", mean)
    print("Scaler std:", std)

    df = df.sort_values(
        ["t0_timestamp", "node_id"]
    )

    groups = df.groupby(
        "t0_timestamp",
        sort=False,
    )

    selected_indices = sorted(set([
        0,
        len(X_train) // 2,
        len(X_train) - 1,
    ]))

    selected_groups = {}

    for index, (_, group) in enumerate(groups):
        if index in selected_indices:
            selected_groups[index] = group

        if len(selected_groups) == len(selected_indices):
            break

    assert len(selected_groups) == len(selected_indices), (
        "Không tìm đủ các sample cần kiểm tra"
    )

    for index in selected_indices:

        expected_X, expected_Y = build_reference_sample(
            selected_groups[index],
            mean,
            std,
        )

        actual_X = X_train[index]
        actual_Y = Y_train[index]

        np.testing.assert_allclose(
            actual_X,
            expected_X,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"X_train sample {index} không khớp",
        )

        np.testing.assert_allclose(
            actual_Y,
            expected_Y,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"Y_train sample {index} không khớp",
        )

        print(f"Sample {index}: PASS")

    print("PASS: X/Y khớp dữ liệu Parquet sau normalization")


def test_normalization(data):
    """Kiểm tra riêng speed và time-of-day."""

    X = data["train_x"]
    Y = data["train_y"]

    speed = X[..., 0]
    time_of_day = X[..., 1]

    print("\n========== 5. NORMALIZATION ==========")

    print("\nSpeed:")
    print("Mean:", speed.mean())
    print("Std:", speed.std())

    print("\nTime of day:")
    print("Mean:", time_of_day.mean())
    print("Std:", time_of_day.std())

    print("\nY speed:")
    print("Mean:", Y.mean())
    print("Std:", Y.std())

    assert np.isfinite(X).all()
    assert np.isfinite(Y).all()

    # Speed X phải được chuẩn hóa bằng scaler Train.
    assert np.isclose(speed.mean(), 0, atol=1e-4)
    assert np.isclose(speed.std(), 1, atol=1e-4)

    print("\nPASS: Normalization hợp lệ")


def run_all_tests():
    print("Loading processed METR-LA...")

    data = load_metr_la()

    print("Loading raw METR-LA Train...")
    raw_df = load_raw_train()

    test_train_shapes(data)
    test_feature_columns(raw_df)
    test_node_ordering(raw_df)
    test_raw_vs_processed(data, raw_df)
    test_normalization(data)

    print("\n================================")
    print("ALL METR-LA INTEGRITY TESTS PASSED")
    print("================================")


if __name__ == "__main__":
    run_all_tests()
