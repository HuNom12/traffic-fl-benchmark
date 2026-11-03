import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.data.dataset import (
    compute_adjacency_matrix,
    load_and_preprocess_pems,
    StandardTrafficDataset,
)


def test_dataset_pipeline():
    print("=== [Test Task 1.2] PeMS Data Pipeline & Adjacency Matrix ===")

    # 1. Kiểm tra Ma trận kề A
    num_nodes = 20
    adj = compute_adjacency_matrix("data/raw/distance.csv", num_nodes=num_nodes)
    assert adj.shape == (num_nodes, num_nodes), "Kích thước ma trận kề sai lệch!"
    assert np.allclose(adj, adj.T), "Ma trận kề không đối xứng!"
    assert np.allclose(np.diag(adj), 1.0), "Đường chéo chính của A chưa bằng 1.0!"
    print("-> Ma trận kề A (Shape, Đối xứng, Self-loop) [PASSED]")

    # 2. Kiểm tra nạp và tiền xử lý PeMS
    data_dict = load_and_preprocess_pems("data/raw/pems04.npz", time_in=12, time_out=12)
    tx, ty = data_dict["train_x"], data_dict["train_y"]
    vx, vy = data_dict["val_x"], data_dict["val_y"]

    # Kiểm tra sliding window
    assert tx.shape[1] == 12, "Time_in phải đúng 12 bước!"
    assert ty.shape[1] == 12, "Time_out phải đúng 12 bước!"
    assert ty.shape[-1] == 1, "Tensor mục tiêu Y chỉ dự báo kênh 0 (Traffic Flow)!"
    print(f"-> Tensor Train X: {tx.shape}, Train Y: {ty.shape} [PASSED]")

    # 3. Kiểm tra PyTorch Dataset
    dataset = StandardTrafficDataset(tx[:32], ty[:32])
    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (12, tx.shape[2], tx.shape[3])
    print("-> StandardTrafficDataset PyTorch wrapping [PASSED]")
    print(">>> TASK 1.2 PASS HOÀN TOÀN <<<\n")


if __name__ == "__main__":
    test_dataset_pipeline()