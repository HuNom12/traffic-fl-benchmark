import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated.client import ClientNode
from src.federated.server import FederatedServer
from src.models.base import BaseForecaster
from src.utils.seed import set_seed


class DummyForecaster(BaseForecaster):
    def __init__(self, num_nodes: int, in_features: int, time_in: int, time_out: int):
        super().__init__(num_nodes, in_features, time_in, time_out)
        self.fc = nn.Linear(time_in * in_features, time_out * in_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        b, t, n, f = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(b, n, t * f)
        out = self.fc(x_flat)
        return out.reshape(b, n, self.time_out, self.in_features).permute(0, 2, 1, 3)


def test_core_fl_pipeline():
    print("Core FL Interfaces and Reproducibility")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Kiểm tra tensor shape từ BaseForecaster
    model = DummyForecaster(num_nodes=10, in_features=1, time_in=12, time_out=12).to(device)
    dummy_x = torch.randn(8, 12, 10, 1).to(device)
    dummy_y = torch.randn(8, 12, 10, 1).to(device)
    pred = model(dummy_x)
    assert pred.shape == dummy_y.shape, f"Lỗi shape: {pred.shape} vs {dummy_y.shape}"
    print("[PASSED]")

    # Kiểm tra ClientNode (FedAvg vs FedProx)
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=4)
    client1 = ClientNode(0, model, loader, loader, device, mu=0.0)
    loss1 = client1.train_epoch()
    assert loss1 > 0, "Loss FedAvg không hợp lệ"

    client2 = ClientNode(1, model, loader, loader, device, mu=0.01)
    loss2 = client2.train_epoch(global_parameters=client1.get_parameters())
    assert loss2 > 0, "Loss FedProx không hợp lệ"
    print(f"-> ClientNode (FedAvg loss: {loss1:.4f}, FedProx loss: {loss2:.4f}) [PASSED]")

    # Kiểm tra Server Aggregator
    server = FederatedServer(model, device)
    server.aggregate([client1.get_parameters(), client2.get_parameters()], [4, 4])
    print("-> Server Aggregation [PASSED]")


if __name__ == "__main__":
    test_core_fl_pipeline()