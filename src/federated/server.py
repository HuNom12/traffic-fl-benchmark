from typing import List, Dict
import copy
import torch
from src.models.base import BaseForecaster


class FederatedServer:
    """Server trung tâm điều phối và tổng hợp mô hình toàn cục."""

    def __init__(self, global_model: BaseForecaster, device: torch.device):
        self.global_model = global_model.to(device)
        self.device = device

    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Lấy trọng số toàn cục hiện tại."""
        return {key: val.clone() for key, val in self.global_model.state_dict().items()}

    def aggregate(
        self,
        client_parameters: List[Dict[str, torch.Tensor]],
        sample_counts: List[int],
    ) -> None:
        """Tổng hợp trọng số theo thuật toán Federated Averaging.

        Args:
            client_parameters: Danh sách dict trọng số của các Client gửi lên
            sample_counts: Số lượng mẫu dữ liệu của từng Client để tính trọng số đóng góp
        """
        total_samples = sum(sample_counts)
        aggregated_weights = copy.deepcopy(client_parameters[0])

        for key in aggregated_weights.keys():
            aggregated_weights[key] = (
                client_parameters[0][key].to(self.device) * (sample_counts[0] / total_samples)
            )
            for i in range(1, len(client_parameters)):
                weighted_param = client_parameters[i][key].to(self.device) * (
                    sample_counts[i] / total_samples
                )
                aggregated_weights[key] += weighted_param

        self.global_model.load_state_dict(aggregated_weights)   