import copy
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.temporal import GRUForecaster
from src.federated.client import ClientNode
from src.federated.server import FederatedServer


def _create_dummy_loader(num_samples: int = 16, batch_size: int = 8, num_nodes: int = 10) -> DataLoader:
    """Tạo DataLoader giả lập đúng chuẩn 4D của METR-LA: X=(B, 12, N, 2), Y=(B, 12, N, 1)."""
    x = torch.randn(num_samples, 12, num_nodes, 2)
    y = torch.randn(num_samples, 12, num_nodes, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_client_node_train_and_fedprox():
    """Kiểm tra ClientNode.train() cho cả FedAvg (mu=0.0) và FedProx (mu>0.0)."""
    device = torch.device("cpu")
    num_nodes = 10
    loader = _create_dummy_loader(num_samples=16, batch_size=8, num_nodes=num_nodes)

    model = GRUForecaster(num_nodes=num_nodes, in_features=2, hidden_dim=16, num_layers=1)
    initial_weights = copy.deepcopy(model.state_dict())

    # 1. Test FedAvg (mu = 0.0)
    client_avg = ClientNode(
        client_id=0,
        model=model,
        train_loader=loader,
        device=device,
        epochs=1,
        lr=1e-3,
        mu=0.0,
    )
    weights_avg, n_samples = client_avg.train()

    assert n_samples == 16
    assert isinstance(weights_avg, dict)
    assert any(not torch.equal(initial_weights[k], weights_avg[k]) for k in initial_weights)

    # 2. Test FedProx (mu = 0.01)
    client_prox = ClientNode(
        client_id=1,
        model=copy.deepcopy(model),
        train_loader=loader,
        device=device,
        epochs=1,
        lr=1e-3,
        mu=0.01,
    )
    client_prox.set_weights(initial_weights)
    weights_prox, n_samples_prox = client_prox.train()

    assert n_samples_prox == 16
    assert isinstance(weights_prox, dict)


def test_federated_server_aggregation_and_sync():
    """Kiểm tra FederatedServer.aggregate() với list các tuple (weights, n_samples) và đồng bộ xuống Client."""
    device = torch.device("cpu")
    num_nodes = 10

    global_model = GRUForecaster(num_nodes=num_nodes, in_features=2, hidden_dim=16, num_layers=1)
    server = FederatedServer(global_model=global_model, device=device)

    loader1 = _create_dummy_loader(num_samples=16, batch_size=8, num_nodes=num_nodes)
    loader2 = _create_dummy_loader(num_samples=32, batch_size=8, num_nodes=num_nodes)

    client1 = ClientNode(
        client_id=0,
        model=copy.deepcopy(global_model),
        train_loader=loader1,
        device=device,
        epochs=1,
        lr=1e-3,
        mu=0.0,
    )
    client2 = ClientNode(
        client_id=1,
        model=copy.deepcopy(global_model),
        train_loader=loader2,
        device=device,
        epochs=1,
        lr=1e-3,
        mu=0.0,
    )

    # Đồng bộ trọng số từ Server xuống Client trước vòng giao tiếp
    global_weights = server.get_global_weights()
    client1.set_weights(global_weights)
    client2.set_weights(global_weights)

    # Huấn luyện cục bộ tại từng Client
    weights1, n_samples1 = client1.train()
    weights2, n_samples2 = client2.train()

    # Tổng hợp tại Server theo đúng chữ ký List[Tuple[Dict, int]]
    server.aggregate([(weights1, n_samples1), (weights2, n_samples2)])
    updated_global_weights = server.get_global_weights()

    # Kiểm tra công thức trung bình có trọng số (16/48 và 32/48)
    for key in updated_global_weights:
        expected = (weights1[key] * 16 + weights2[key] * 32) / 48.0
        assert torch.allclose(updated_global_weights[key], expected, atol=1e-5)

    # Cập nhật ngược lại cho Client1 và kiểm tra khớp hoàn toàn
    client1.set_weights(updated_global_weights)
    for key, val in client1.model.state_dict().items():
        assert torch.equal(val, updated_global_weights[key])