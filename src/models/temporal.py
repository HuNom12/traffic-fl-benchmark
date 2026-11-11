from typing import Optional
import torch
import torch.nn as nn

from src.models.base import BaseForecaster


class _RNNForecaster(BaseForecaster):
    """Lớp nền dùng chung cho các mô hình chuỗi thời gian thuần túy (Spatial-agnostic).

    Chia sẻ trọng số mạng hồi quy qua tất cả các nút cảm biến độc lập mà không
    sử dụng thông tin cấu trúc đồ thị (ma trận kề A).
    """

    def __init__(
        self,
        rnn_type: str,
        num_nodes: int,
        in_features: int = 2,
        time_in: int = 12,
        time_out: int = 12,
        out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(num_nodes, in_features, time_in, time_out)
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_out * out_features),
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Thực hiện dự báo chuỗi thời gian từ đặc trưng lịch sử.

        Args:
            x: Tensor đầu vào có kích thước (Batch, Time_in, Nodes, In_features)
            adj: Ma trận kề đồ thị (được bỏ qua với mô hình baseline thuần thời gian)

        Returns:
            Tensor dự báo có kích thước (Batch, Time_out, Nodes, Out_features)
        """
        b, t_in, n, f_in = x.shape

        # Biến đổi về (Batch * Nodes, Time_in, In_features) để xử lý độc lập từng node
        x_seq = x.permute(0, 2, 1, 3).reshape(b * n, t_in, f_in)

        rnn_out, _ = self.rnn(x_seq)  # (B * N, T_in, hidden_dim)
        last_hidden = rnn_out[:, -1, :]  # Lấy trạng thái ẩn bước cuối: (B * N, hidden_dim)

        out = self.head(last_hidden)  # (B * N, time_out * out_features)
        out = out.view(b, n, self.time_out, self.out_features)
        return out.permute(0, 2, 1, 3).contiguous()


class GRUForecaster(_RNNForecaster):
    """Mô hình baseline chuỗi thời gian sử dụng kiến trúc 2 lớp GRU."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int = 2,
        time_in: int = 12,
        time_out: int = 12,
        out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(
            rnn_type="GRU",
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )


class LSTMForecaster(_RNNForecaster):
    """Mô hình baseline chuỗi thời gian sử dụng kiến trúc 2 lớp LSTM."""

    def __init__(
        self,
        num_nodes: int,
        in_features: int = 2,
        time_in: int = 12,
        time_out: int = 12,
        out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(
            rnn_type="LSTM",
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

