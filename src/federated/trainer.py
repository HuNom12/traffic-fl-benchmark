import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional

from src.federated.client import ClientNode
from src.federated.server import FederatedServer

class FederatedTrainer:
    """Engine điều phối quá trình huấn luyện mô hình dự báo giao thông."""

    def __init__(
        self,
        model_builder,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model_builder = model_builder
        self.device = device
        self.config = config

    def train_centralized(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        """Huấn luyện tập trung (Centralized Learning - Upper-bound Benchmark)."""
        model = self.model_builder().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get("lr", 0.001))
        criterion = nn.MSELoss()
        epochs = self.config.get("epochs", 10)

        print("\n--- Bắt đầu huấn luyện Centralized Learning ---")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * batch_x.size(0)
            
            epoch_loss = total_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

        return model

    def train_local(self, client_loaders: List[DataLoader], epochs: Optional[int] = None) -> List[nn.Module]:
        """Huấn luyện độc lập tại từng Client (Local Learning - Lower-bound Benchmark).
        
        Mỗi client tự train mô hình riêng trên dữ liệu của mình, không chia sẻ trọng số.
        """
        local_epochs = epochs if epochs is not None else self.config.get("epochs", 10)
        lr = self.config.get("lr", 0.001)
        
        trained_models = []
        
        print("\n--- Bắt đầu huấn luyện Local Learning (Lower-bound) ---")
        for i, loader in enumerate(client_loaders):
            print(f"Đang huấn luyện Client độc lập [{i+1}/{len(client_loaders)}]...")
            
            # Khởi tạo mô hình riêng cho client này
            client_model = self.model_builder().to(self.device)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            client_model.train()
            for epoch in range(local_epochs):
                total_loss = 0.0
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    pred = client_model(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item() * batch_x.size(0)
                
                epoch_loss = total_loss / len(loader.dataset)
                print(f"  Client {i} - Epoch [{epoch+1}/{local_epochs}] - Loss: {epoch_loss:.4f}")
                
            trained_models.append(client_model)
            
        return trained_models

    def train_federated(
        self,
        client_loaders: List[DataLoader],
        val_loader: DataLoader
    ) -> nn.Module:
        """Huấn luyện liên kết (Federated Learning / FedAvg / FedProx)."""
        rounds = self.config.get("rounds", 10)
        local_epochs = self.config.get("local_epochs", 1)
        lr = self.config.get("lr", 0.001)
        mu = self.config.get("mu", 0.0)

        # Khởi tạo Global Model và Server
        global_model = self.model_builder()
        server = FederatedServer(global_model, self.device)

        # Khởi tạo danh sách Clients
        clients = [
            ClientNode(
                client_id=i,
                model=self.model_builder(),
                train_loader=loader,
                device=self.device,
                epochs=local_epochs,
                lr=lr,
                mu=mu
            )
            for i, loader in enumerate(client_loaders)
        ]

        print(f"\n--- Bắt đầu huấn luyện Federated Learning ({rounds} rounds) ---")
        for r in range(rounds):
            print(f"\n[Communication Round {r+1}/{rounds}]")
            
            global_weights = server.get_global_weights()
            client_results = []

            # Huấn luyện song song/lần lượt trên các client
            for client in clients:
                client.set_weights(global_weights)
                weights, num_samples = client.train()
                client_results.append((weights, num_samples))

            # Server tổng hợp trọng số
            server.aggregate(client_results)

        return server.global_model