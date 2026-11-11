import os
import sys

# Đảm bảo Python nhận diện được package src khi chạy trực tiếp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed
from src.models import (
    BaseForecaster,
    GRUForecaster,
    LSTMForecaster,
    STGCNForecaster,
    build_model,
)
from src.federated.client import ClientNode
from src.federated.server import FederatedServer


def _make_symmetric_adj(num_nodes: int) -> torch.Tensor:
    """Sinh ma trận kề đối xứng giả lập có đường chéo chính bằng 1."""
    raw = torch.rand(num_nodes, num_nodes)
    adj = (raw + raw.T) / 2.0
    adj = torch.where(adj > 0.7, adj, torch.zeros_like(adj))
    adj.fill_diagonal_(1.0)
    return adj


def _assert_gradients_flow(model: nn.Module, pred: torch.Tensor, target: torch.Tensor) -> None:
    """Kiểm tra lan truyền ngược (backward pass) tạo gradient hợp lệ cho toàn bộ tham số."""
    model.zero_grad()
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Tham số {name} không có gradient"
            assert not torch.isnan(param.grad).any(), f"Gradient của {name} chứa NaN"


def test_temporal_baselines():
    set_seed(42)
    batch_size, time_in, num_nodes, in_features = 8, 12, 207, 2
    time_out, out_features = 12, 1

    x = torch.randn(batch_size, time_in, num_nodes, in_features)
    y = torch.randn(batch_size, time_out, num_nodes, out_features)
    adj_global = _make_symmetric_adj(num_nodes)

    for cls_name, model_cls in [("GRU", GRUForecaster), ("LSTM", LSTMForecaster)]:
        model = model_cls(
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
        )
        assert isinstance(model, BaseForecaster)

        # Kiểm tra khi không truyền adj và khi truyền adj (đều cho output đúng shape)
        pred = model(x)
        assert pred.shape == (batch_size, time_out, num_nodes, out_features), (
            f"{cls_name} output shape sai: {pred.shape}"
        )
        pred_with_adj = model(x, adj=adj_global)
        assert pred_with_adj.shape == (batch_size, time_out, num_nodes, out_features)

        # Kiểm tra trên đồ thị con N_k = 45 (Client subgraph)
        x_sub = torch.randn(batch_size, time_in, 45, in_features)
        pred_sub = model(x_sub)
        assert pred_sub.shape == (batch_size, time_out, 45, out_features)

        _assert_gradients_flow(model, pred, y)
        print(f"-> {cls_name}Forecaster forward (global N=207 & subgraph N_k=45) + backward [PASSED]")


def test_stgcn_forecaster():
    set_seed(42)
    batch_size, time_in, num_nodes, in_features = 8, 12, 207, 2
    time_out, out_features = 12, 1

    x_global = torch.randn(batch_size, time_in, num_nodes, in_features)
    y_global = torch.randn(batch_size, time_out, num_nodes, out_features)
    adj_global = _make_symmetric_adj(num_nodes)

    for conv_type in ["cheb", "gcn"]:
        model = STGCNForecaster(
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            conv_type=conv_type,
        )
        assert isinstance(model, BaseForecaster)

        # 1. Forward với ma trận toàn cục 207x207
        pred_global = model(x_global, adj=adj_global)
        assert pred_global.shape == (batch_size, time_out, num_nodes, out_features), (
            f"STGCN ({conv_type}) global shape sai: {pred_global.shape}"
        )

        # 2. Forward với đồ thị con cục bộ của Client (N_k = 45)
        n_sub = 45
        x_sub = torch.randn(batch_size, time_in, n_sub, in_features)
        adj_sub = _make_symmetric_adj(n_sub)
        pred_sub = model(x_sub, adj=adj_sub)
        assert pred_sub.shape == (batch_size, time_out, n_sub, out_features), (
            f"STGCN ({conv_type}) subgraph shape sai: {pred_sub.shape}"
        )

        # 3. Forward khi adj=None (fallback an toàn)
        pred_no_adj = model(x_sub)
        assert pred_no_adj.shape == (batch_size, time_out, n_sub, out_features)

        # 4. Kiểm tra Gradient
        _assert_gradients_flow(model, pred_global, y_global)
        print(f"-> STGCNForecaster (conv_type='{conv_type}') global 207x207 + subgraph 45x45 + backward [PASSED]")


def test_federated_compatibility_with_heterogeneous_subgraphs():
    """Kiểm tra STGCN hoạt động trơn tru với ClientNode & FederatedServer ngay cả khi
    các Client sở hữu đồ thị con có số nút N_k khác nhau (45 vs 60 vs 207).
    """
    set_seed(42)
    device = torch.device("cpu")
    time_in, time_out, in_features, out_features = 12, 12, 2, 1

    # Client 0: 45 nodes, Client 1: 60 nodes, Server Global: 207 nodes
    adj_c0 = _make_symmetric_adj(45)
    adj_c1 = _make_symmetric_adj(60)
    adj_global = _make_symmetric_adj(207)

    x_c0 = torch.randn(8, time_in, 45, in_features)
    y_c0 = torch.randn(8, time_out, 45, out_features)
    x_c1 = torch.randn(8, time_in, 60, in_features)
    y_c1 = torch.randn(8, time_out, 60, out_features)

    loader_c0 = DataLoader(TensorDataset(x_c0, y_c0), batch_size=4)
    loader_c1 = DataLoader(TensorDataset(x_c1, y_c1), batch_size=4)

    model_server = STGCNForecaster(num_nodes=207, default_adj=adj_global)
    model_c0 = STGCNForecaster(num_nodes=45, default_adj=adj_c0)
    model_c1 = STGCNForecaster(num_nodes=60, default_adj=adj_c1)

    server = FederatedServer(model_server, device=device)
    client0 = ClientNode(0, model_c0, loader_c0, loader_c0, device=device, mu=0.0)
    client1 = ClientNode(1, model_c1, loader_c1, loader_c1, device=device, mu=0.01)

    global_params = server.get_global_parameters()
    client0.set_parameters(global_params)
    client1.set_parameters(global_params)

    loss0 = client0.train_epoch()
    loss1 = client1.train_epoch(global_parameters=global_params)

    # Server tổng hợp trọng số từ 2 client có số nút khác nhau
    server.aggregate([client0.get_parameters(), client1.get_parameters()], [8, 8])

    # Đánh giá thử trên toàn bộ 207 nodes
    x_test = torch.randn(4, time_in, 207, in_features)
    with torch.no_grad():
        pred_global = server.global_model(x_test, adj=adj_global)
    assert pred_global.shape == (4, time_out, 207, out_features)
    print(
        f"-> FL Heterogeneous Subgraphs (Client0 N=45 loss={loss0:.4f}, "
        f"Client1 N=60 loss={loss1:.4f} -> Global N=207) [PASSED]"
    )


def test_model_factory():
    cfg = {
        "dataset": {"num_nodes": 207, "features": 2, "time_in": 12, "time_out": 12},
        "model": {"name": "STGCN", "hidden_dim": 64, "num_layers": 2, "dropout": 0.2},
    }
    stgcn = build_model(cfg)
    assert isinstance(stgcn, STGCNForecaster)

    cfg["model"]["name"] = "GRU"
    gru = build_model(cfg)
    assert isinstance(gru, GRUForecaster)

    cfg["model"]["name"] = "LSTM"
    lstm = build_model(cfg)
    assert isinstance(lstm, LSTMForecaster)
    print("-> build_model factory (STGCN, GRU, LSTM) [PASSED]")


if __name__ == "__main__":
    print("=== [Test Task 2.2] Kiểm thử Mô hình Spatio-Temporal (STGCN) & Temporal Baselines (GRU/LSTM) ===")
    test_temporal_baselines()
    test_stgcn_forecaster()
    test_federated_compatibility_with_heterogeneous_subgraphs()
    test_model_factory()
    print("\n>>> TẤT CẢ UNIT TESTS CHO TASK 2.2 ĐÃ HOÀN THÀNH XUẤT SẮC! <<<")

