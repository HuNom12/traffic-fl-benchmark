import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.federated.trainer import FederatedTrainer

# 1. Định nghĩa một mô hình giả lập đơn giản kế thừa nn.Module
class DummyTrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Giả lập đầu vào flatten từ shape (Batch, 12, 207, 2) hoặc đơn giản là Linear
        self.fc = nn.Linear(12 * 207 * 2, 12 * 207 * 1)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        out = self.fc(x_flat)
        # Reshape lại đúng dạng output mong đợi (Batch, 12, 207, 1)
        return out.reshape(batch_size, 12, 207, 1)

def test_fl_engine_pipeline():
    print("=== BẮT ĐẦU CHẠY KIỂM THỬ FL ENGINE ===")
    
    device = torch.device("cpu")
    
    # 2. Tạo Mock Data giả lập khớp với chiều tensor của dự án
    # Batch size = 16, seq_len = 12, nodes = 207, features = 2
    mock_x = torch.randn(16, 12, 207, 2)
    # Target shape: (Batch = 16, pred_len = 12, nodes = 207, features = 1)
    mock_y = torch.randn(16, 12, 207, 1)
    
    dataset = TensorDataset(mock_x, mock_y)
    loader = DataLoader(dataset, batch_size=8)
    
    client_loaders = [loader, loader] # Giả lập 2 Clients
    
    # 3. Cấu hình huấn luyện
    config = {
        "epochs": 2,
        "rounds": 2,
        "local_epochs": 1,
        "lr": 0.01,
        "mu": 0.1  # Test cả FedProx
    }
    
    trainer = FederatedTrainer(
        model_builder=DummyTrafficModel,
        device=device,
        config=config
    )
    
    # 4. Kiểm tra chế độ Centralized Learning
    print("\n[Test 1] Đang kiểm tra Centralized Learning...")
    centralized_model = trainer.train_centralized(loader, loader)
    assert isinstance(centralized_model, nn.Module)
    print("-> Centralized Learning: PASS")
    
    # 5. Kiểm tra chế độ Local Learning
    print("\n[Test 2] Đang kiểm tra Local Learning...")
    local_models = trainer.train_local(client_loaders, epochs=1)
    assert len(local_models) == 2
    print("-> Local Learning: PASS")
    
    # 6. Kiểm tra chế độ Federated Learning (FedAvg / FedProx)
    print("\n[Test 3] Đang kiểm tra Federated Learning (FedProx)...")
    global_model = trainer.train_federated(client_loaders, loader)
    assert isinstance(global_model, nn.Module)
    print("-> Federated Learning: PASS")
    
    print("\n=== TẤT CẢ CÁC KIỂM THỬ FL ENGINE ĐÃ THÀNH CÔNG! ===")

if __name__ == "__main__":
    test_fl_engine_pipeline()