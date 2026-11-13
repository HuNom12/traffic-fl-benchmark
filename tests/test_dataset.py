
import numpy as np

from src.data.dataset import load_metr_la


def test_display_train_data():
    """Load và hiển thị dữ liệu X_train, Y_train."""

    data = load_metr_la()

    X_train = data["train_x"]
    Y_train = data["train_y"]

    print("\n========== TRAIN DATA ==========")

    # 1. Kiểm tra shape
    print("\n[SHAPE]")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)

    # 2. Kiểm tra dtype
    print("\n[DTYPE]")
    print("X_train:", X_train.dtype)
    print("Y_train:", Y_train.dtype)

    # 3. Hiển thị một sample
    print("\n[SAMPLE 0]")
    print("X_train[0]:")
    print(X_train[0])

    print("\nY_train[0]:")
    print(Y_train[0])

    # 4. Hiển thị một timestep
    print("\n[TIMESTEP 0]")
    print("X_train[0, 0] - 207 nodes, 2 features:")
    print(X_train[0, 0])

    print("\nY_train[0, 0] - 207 nodes, 1 target:")
    print(Y_train[0, 0])

    # 5. Thống kê dữ liệu
    print("\n[STATISTICS]")
    print(
        f"X_train - min: {X_train.min():.4f}, "
        f"max: {X_train.max():.4f}, "
        f"mean: {X_train.mean():.4f}, "
        f"std: {X_train.std():.4f}"
    )

    print(
        f"Y_train - min: {Y_train.min():.4f}, "
        f"max: {Y_train.max():.4f}, "
        f"mean: {Y_train.mean():.4f}, "
        f"std: {Y_train.std():.4f}"
    )

    # 6. Kiểm tra dữ liệu
    assert X_train.ndim == 4
    assert Y_train.ndim == 4

    assert X_train.shape[1:] == (12, 207, 2)
    assert Y_train.shape[1:] == (12, 207, 1)

    assert np.isfinite(X_train).all()
    assert np.isfinite(Y_train).all()

    print("\nDataset test passed!")


if __name__ == "__main__":
    test_display_train_data()
