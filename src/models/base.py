from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseForecaster(nn.Module, ABC):
    def __init__(self, num_nodes: int, in_features: int, time_in: int, time_out: int):
        super(BaseForecaster, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.time_in = time_in
        self.time_out = time_out

    @abstractmethod
    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        """Thực hiện dự báo từ chuỗi lịch sử.

        Args:
            x: Tensor đầu vào có kích thước (Batch, Time_in, Nodes, Features)
            adj: Ma trận kề đồ thị (tùy chọn với mô hình đồ thị, bỏ qua với GRU/LSTM)

        Returns:
            Tensor dự báo có kích thước (Batch, Time_out, Nodes, Features)
        """
        pass