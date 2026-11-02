from typing import Dict
import copy
import torch
from torch.utils.data import DataLoader
from src.models.base import BaseForecaster


class ClientNode:
    """Đại diện cho một Client/Node trong hệ thống phân tán."""

    def __init__(
        self,
        client_id: int,
        model: BaseForecaster,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        mu: float = 0.0,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mu = mu  # Tham số proximal term cho FedProx (mu=0 tương đương FedAvg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def set_parameters(self, global_parameters: Dict[str, torch.Tensor]) -> None:
        """Cập nhật trọng số từ Server gửi về."""
        self.model.load_state_dict(global_parameters)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Lấy trọng số cục bộ để gửi về Server."""
        return {key: val.cpu().clone() for key, val in self.model.state_dict().items()}

    def train_epoch(self, global_parameters: Dict[str, torch.Tensor] = None) -> float:
        """Huấn luyện 1 local epoch, hỗ trợ cả FedAvg và FedProx."""
        self.model.train()
        total_loss = 0.0

        # Lưu lại bản sao trọng số toàn cục nếu chạy FedProx
        global_weight_dict = None
        if self.mu > 0.0 and global_parameters is not None:
            global_weight_dict = {
                k: v.to(self.device) for k, v in global_parameters.items()
            }

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(x)
            loss = self.criterion(pred, y)

            # Thêm Proximal term nếu dùng FedProx: (mu / 2) * ||w - w_global||^2
            if self.mu > 0.0 and global_weight_dict is not None:
                prox_term = 0.0
                for name, param in self.model.named_parameters():
                    prox_term += torch.sum((param - global_weight_dict[name]) ** 2)
                loss += (self.mu / 2.0) * prox_term

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)