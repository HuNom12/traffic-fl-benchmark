import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import copy

class FederatedServer:
    """Điều phối trung tâm cho thuật toán Federated Learning (Server)."""

    def __init__(self, global_model: nn.Module, device: torch.device):
        self.global_model = global_model.to(device)
        self.device = device

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Lấy trọng số hiện tại của mô hình toàn cục."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_results: List[Tuple[Dict[str, torch.Tensor], int]]) -> None:
        """Thực hiện tổng hợp FedAvg dựa trên trọng số số lượng mẫu dữ liệu từ các Client.

        Args:
            client_results: Danh sách các tuple (client_weights, num_samples)
        """
        if not client_results:
            return

        total_samples = sum(num_samples for _, num_samples in client_results)
        if total_samples == 0:
            return

        # Khởi tạo accumulator cho trọng số mới
        global_weights = self.global_model.state_dict()
        for key in global_weights:
            global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)

        # Tính tổng trọng số có trọng số (weighted average)
        for client_weights, num_samples in client_results:
            weight_factor = num_samples / total_samples
            for key in global_weights:
                global_weights[key] += client_weights[key].float() * weight_factor

        # Cập nhật vào mô hình toàn cục
        self.global_model.load_state_dict(global_weights)