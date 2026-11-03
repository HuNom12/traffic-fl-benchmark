import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed
from src.models.base import BaseForecaster
from src.federated.client import ClientNode
from src.federated.server import FederatedServer
from src.data.partition import (
    partition_nodes_spatial_clustering,
    generate_spatial_skew_scenario,
    generate_temporal_shift_scenario,
)


# 1. Tạo một Dummy Model đơn giản kế thừa BaseForecaster để test khung interface
class DummyForecaster(BaseForecaster):
    def __init__(self, num_nodes: int, in_features: int, time_in: int, time_out: int):
        super().__init__(num_nodes, in_features, time_in, time_out)
        # Linear layer biến đổi từ time_in -> time_out
        self.fc = nn.Linear(time_in * in_features, time_out * in_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        # x: (Batch, Time_in, Nodes, Features)
        b, t, n, f = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(b, n, t * f)
        out = self.fc(x_flat)
        out = out.reshape(b, n, self.time_out, self.in_features).permute(0, 2, 1, 3)
        return out


def run_unit_test():
    print("=== [1] Kiểm tra cố định Seed ===")
    set_seed(42)
    print("-> set_seed(42) chạy thành công!")

    # Cấu hình tham số giả lập
    batch_size = 8
    time_in = 12
    time_out = 12
    num_nodes = 10
    features = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Đang chạy test trên thiết bị: {device}")

    print("\n=== [2] Kiểm tra BaseForecaster & Shape Tensor ===")
    model = DummyForecaster(num_nodes, features, time_in, time_out).to(device)
    dummy_x = torch.randn(batch_size, time_in, num_nodes, features).to(device)
    dummy_y = torch.randn(batch_size, time_out, num_nodes, features).to(device)
    pred = model(dummy_x)
    assert pred.shape == dummy_y.shape, f"Lỗi shape. Kỳ vọng {dummy_y.shape}, nhận {pred.shape}"
    print(f"-> Forward thành công, output shape hợp lệ: {pred.shape}")

    print("\n=== [3] Kiểm tra Luồng Huấn luyện ClientNode (FedAvg & FedProx) ===")
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=4)

    # Khởi tạo 2 client
    model_c1 = DummyForecaster(num_nodes, features, time_in, time_out)
    model_c2 = DummyForecaster(num_nodes, features, time_in, time_out)
    client1 = ClientNode(client_id=0, model=model_c1, train_loader=loader, val_loader=loader, device=device, mu=0.0)
    client2 = ClientNode(client_id=1, model=model_c2, train_loader=loader, val_loader=loader, device=device, mu=0.01)

    loss1 = client1.train_epoch()
    print(f"-> Client 1 (FedAvg - mu=0.0) huấn luyện 1 epoch, Loss: {loss1:.4f}")

    global_weights = client1.get_parameters()
    loss2 = client2.train_epoch(global_parameters=global_weights)
    print(f"-> Client 2 (FedProx - mu=0.01) huấn luyện 1 epoch với proximal term, Loss: {loss2:.4f}")

    print("\n=== [4] Kiểm tra FederatedServer & Hàm Aggregate ===")
    server_model = DummyForecaster(num_nodes, features, time_in, time_out)
    server = FederatedServer(global_model=server_model, device=device)

    # Server gom weights từ 2 client và aggregate
    client_weights = [client1.get_parameters(), client2.get_parameters()]
    sample_counts = [len(loader.dataset), len(loader.dataset)]
    server.aggregate(client_weights, sample_counts)
    print("-> Server aggregate FedAvg thành công.")

    print("\n=== [5] Kiểm tra Phân chia Dữ liệu Non-IID (Task 1.3) ===")
    adj = np.eye(num_nodes)
    clusters = partition_nodes_spatial_clustering(adj, num_clients=2)
    print(f"-> Phân chia cụm không gian thành công cho 2 client: {clusters}")

    dummy_traffic = np.random.rand(100, num_nodes)
    res_spatial = generate_spatial_skew_scenario(dummy_traffic, clusters, output_path="splits/spatial_skew.json")
    print(f"-> Sinh kịch bản Spatial Skew thành công, clients: {list(res_spatial['clients'].keys())}")

    res_temporal = generate_temporal_shift_scenario(num_samples=288, num_clients=2, output_path="splits/temporal_shift.json", num_nodes=num_nodes)
    print(f"-> Sinh kịch bản Temporal Shift thành công, clients: {list(res_temporal['clients'].keys())}")

    print("\n>>> Toàn bộ pipeline hoạt động ổn định và chính xác! <<<")


if __name__ == "__main__":
    run_unit_test()
