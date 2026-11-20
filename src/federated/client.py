import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from typing import Dict, Any

class ClientNode:
    """Đại diện cho một Client trong hệ thống Federated Learning."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        epochs: int = 1,
        lr: float = 0.001,
        mu: float = 0.0  # Hệ số FedProx (nếu mu = 0 thì tương đương FedAvg)
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.mu = mu

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def set_weights(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Cập nhật trọng số mô hình cục bộ từ trọng số toàn cục của Server."""
        self.model.load_state_dict(global_weights)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Lấy trọng số hiện tại của mô hình cục bộ."""
        return copy.deepcopy(self.model.state_dict())

    def train_epoch(self, global_parameters: Dict[str, torch.Tensor] = None) -> tuple[Dict[str, torch.Tensor], int]:
        """Thực hiện huấn luyện trong một epoch trên dữ liệu của Client.

        Args:
            global_parameters: Trọng số toàn cục (dùng cho FedProx)

        Returns:
            Tuple gồm (trọng số mô hình sau khi train, tổng số mẫu huấn luyện n_k)
        """
        if global_parameters is not None:
            self.set_weights(global_parameters)
            # Khi có global_parameters, gọi train với mu > 0 nếu mu chưa được thiết lập
            if self.mu == 0.0:
                self.mu = 0.01  # Sử dụng giá trị mặc định cho FedProx
        
        return self.train()

    def train(self) -> tuple[Dict[str, torch.Tensor], int]:
        """Thực hiện huấn luyện cục bộ trên dữ liệu của Client.

        Returns:
            Tuple gồm (trọng số mô hình sau khi train, tổng số mẫu huấn luyện n_k)
        """
        self.model.train()
        
        # Lưu bản sao trọng số toàn cục để tính toán Proximal Term (cho FedProx)
        global_model_params = None
        if self.mu > 0:
            global_model_params = [
                param.clone().detach() for param in self.model.parameters()
            ]

        total_samples = 0

        for epoch in range(self.epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                batch_size = batch_x.size(0)
                total_samples += batch_size

                self.optimizer.zero_grad()
                
                # Dự báo từ mô hình
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                # Bổ sung Proximal term nếu chạy FedProx
                if self.mu > 0 and global_model_params is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_model_params):
                        proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss += (self.mu / 2.0) * proximal_term

                loss.backward()
                self.optimizer.step()

        # Trả về state_dict và số lượng mẫu của client
        return copy.deepcopy(self.model.state_dict()), total_samples