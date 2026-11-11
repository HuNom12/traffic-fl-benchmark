from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseForecaster


def _normalize_adj_gcn(adj: torch.Tensor) -> torch.Tensor:
    """Tính ma trận kề chuẩn hóa đối xứng cho phép chập GCN bậc 1:
    A_norm = D^{-1/2} (A + I_N) D^{-1/2}.
    """
    num_nodes = adj.shape[0]
    a_sym = (adj + adj.T) / 2.0
    a_sym = torch.clamp(a_sym, min=0.0)
    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    a_hat = a_sym + eye
    # Tránh cộng đôi đường chéo nếu adj đã có self-loop sẵn = 1
    diag_mask = torch.diag(a_sym) > 0
    if torch.any(diag_mask):
        a_hat = a_sym.clone()
        a_hat.fill_diagonal_(1.0)

    deg = a_hat.sum(dim=-1)
    d_inv_sqrt = torch.where(
        deg > 0,
        torch.pow(deg.clamp(min=1e-6), -0.5),
        torch.zeros_like(deg),
    )
    return d_inv_sqrt.unsqueeze(1) * a_hat * d_inv_sqrt.unsqueeze(0)


def _chebyshev_polynomials(adj: torch.Tensor, k_order: int) -> torch.Tensor:
    """Tính các đa thức Chebyshev bậc 0..(K-1) dựa trên ma trận Laplacian chuẩn hóa.

    L_sym = I_N - D^{-1/2} A D^{-1/2}
    L_tilde = (2 / lambda_max) * L_sym - I_N  (với lambda_max = 2.0 -> L_tilde = -D^{-1/2} A D^{-1/2})
    """
    num_nodes = adj.shape[0]
    a_sym = (adj + adj.T) / 2.0
    a_sym = torch.clamp(a_sym, min=0.0)

    deg = a_sym.sum(dim=-1)
    d_inv_sqrt = torch.where(
        deg > 0,
        torch.pow(deg.clamp(min=1e-6), -0.5),
        torch.zeros_like(deg),
    )
    norm_a = d_inv_sqrt.unsqueeze(1) * a_sym * d_inv_sqrt.unsqueeze(0)

    eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    l_tilde = -norm_a

    supports = [eye]
    if k_order > 1:
        supports.append(l_tilde)
    for _ in range(2, k_order):
        t_k = 2.0 * torch.matmul(l_tilde, supports[-1]) - supports[-2]
        supports.append(t_k)

    return torch.stack(supports[:k_order], dim=0)  # (K_s, N, N)


class TemporalConvGLU(nn.Module):
    """Lớp Tích chập Thời gian có cổng (Temporal Gated Conv 1D - GLU).

    Áp dụng Conv2d với kernel (K_t, 1) trên tensor (B, C_in, T, N) để trích xuất
    phụ thuộc thời gian ngắn hạn độc lập theo từng node.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
        )
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, N)
        x_res = self.residual_proj(x) if self.residual_proj is not None else x
        if self.padding == 0 and self.kernel_size > 1:
            # Căn chỉnh chiều dài T khi không dùng padding
            x_res = x_res[:, :, self.kernel_size - 1 :, :]

        conv_out = self.conv(x)  # (B, 2 * C_out, T', N)
        p, q = torch.split(conv_out, self.out_channels, dim=1)
        return p * torch.sigmoid(q) + x_res


class SpatialGraphConv(nn.Module):
    """Lớp Tích chập Đồ thị Không gian (Spatial Graph Convolution).

    Hỗ trợ linh hoạt cả:
    - 'cheb': Xấp xỉ đa thức Chebyshev bậc K_s trên ma trận Laplacian chuẩn hóa.
    - 'gcn':  Phép nhân chập đồ thị chuẩn hóa bậc 1 D^{-1/2}(A+I)D^{-1/2}.
    Trọng số hoàn toàn độc lập với số lượng nút N, cho phép chia sẻ/tổng hợp trên mọi kích thước đồ thị con.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "cheb",
        kernel_size_s: int = 3,
    ):
        super().__init__()
        self.conv_type = conv_type.lower()
        self.kernel_size_s = kernel_size_s if self.conv_type == "cheb" else 1

        self.theta = nn.Conv2d(
            in_channels=self.kernel_size_s * in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, N), adj: (N, N)
        b, c_in, t, n = x.shape

        if self.conv_type == "cheb":
            supports = _chebyshev_polynomials(adj, self.kernel_size_s)  # (K_s, N, N)
            # Nhân chập đồ thị qua mọi bậc Chebyshev: (B, K_s, C_in, T, N)
            x_graph = torch.einsum("knm,bctm->bkctn", supports, x)
            x_graph = x_graph.reshape(b, self.kernel_size_s * c_in, t, n)
        else:
            a_norm = _normalize_adj_gcn(adj)  # (N, N)
            x_graph = torch.einsum("nm,bctm->bctn", a_norm, x)

        out = self.theta(x_graph)
        return F.relu(out)


class STConvBlock(nn.Module):
    """Khối Spatio-Temporal Convolutional Block chuẩn dạng Sandwich:
    Temporal Gated Conv (GLU) -> Spatial Graph Conv -> Temporal Gated Conv (GLU).
    """

    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        out_channels: int,
        kernel_size_t: int = 3,
        kernel_size_s: int = 3,
        padding_t: int = 0,
        conv_type: str = "cheb",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tconv1 = TemporalConvGLU(
            in_channels=in_channels,
            out_channels=spatial_channels,
            kernel_size=kernel_size_t,
            padding=padding_t,
        )
        self.sconv = SpatialGraphConv(
            in_channels=spatial_channels,
            out_channels=spatial_channels,
            conv_type=conv_type,
            kernel_size_s=kernel_size_s,
        )
        self.tconv2 = TemporalConvGLU(
            in_channels=spatial_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_t,
            padding=padding_t,
        )
        # LayerNorm chỉ chuẩn hóa trên chiều cuối C -> độc lập hoàn toàn với số lượng nút N
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T, N)
        h = self.tconv1(x)
        h = self.sconv(h, adj)
        h = self.tconv2(h)

        # LayerNorm trên chiều C: (B, C, T, N) -> (B, T, N, C) -> norm -> (B, C, T, N)
        h = self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return self.dropout(h)


class STGCNForecaster(BaseForecaster):
    """Mô hình dự báo không-thời gian STGCN (Spatio-Temporal Graph Convolutional Network).

    Kế thừa BaseForecaster, hỗ trợ cả ma trận kề toàn cục (Centralized: 207x207)
    và ma trận đồ thị con cục bộ của từng Client (Federated/Local: N_k x N_k).
    """

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
        kernel_size_t: int = 3,
        kernel_size_s: int = 3,
        conv_type: str = "cheb",
        default_adj: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        super().__init__(num_nodes, in_features, time_in, time_out)
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type

        # Lưu ma trận kề mặc định dưới dạng thuộc tính thường (không đưa vào state_dict để FedAvg không bị lỗi lệch shape)
        self._default_adj = (
            torch.as_tensor(default_adj, dtype=torch.float32)
            if default_adj is not None
            else None
        )

        # Tính toán độ dài chuỗi thời gian còn lại nếu dùng valid padding (padding_t = 0)
        t_rem = time_in - num_layers * 2 * (kernel_size_t - 1)
        if t_rem >= 1:
            padding_t = 0
            out_kernel_t = t_rem
        else:
            padding_t = (kernel_size_t - 1) // 2
            out_kernel_t = time_in

        spatial_channels = max(hidden_dim // 2, 16)
        self.st_blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            c_in = in_features if layer_idx == 0 else hidden_dim
            self.st_blocks.append(
                STConvBlock(
                    in_channels=c_in,
                    spatial_channels=spatial_channels,
                    out_channels=hidden_dim,
                    kernel_size_t=kernel_size_t,
                    kernel_size_s=kernel_size_s,
                    padding_t=padding_t,
                    conv_type=conv_type,
                    dropout=dropout,
                )
            )

        # Output Block: nén chiều thời gian còn lại về 1 bước rồi chiếu ra (time_out * out_features)
        self.out_tconv = TemporalConvGLU(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=out_kernel_t,
            padding=0,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=time_out * out_features,
                kernel_size=(1, 1),
            ),
        )

    def _resolve_adjacency(
        self, adj: Optional[Union[np.ndarray, torch.Tensor]], num_nodes: int, x: torch.Tensor
    ) -> torch.Tensor:
        """Chuẩn hóa hoặc tạo ma trận kề mặc định khớp với số lượng nút của batch x."""
        if adj is not None:
            adj_tensor = torch.as_tensor(adj, dtype=x.dtype, device=x.device)
        elif self._default_adj is not None and self._default_adj.shape[0] == num_nodes:
            adj_tensor = self._default_adj.to(device=x.device, dtype=x.dtype)
        else:
            adj_tensor = torch.eye(num_nodes, device=x.device, dtype=x.dtype)

        if adj_tensor.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Kích thước ma trận kề {tuple(adj_tensor.shape)} không khớp với số lượng node ({num_nodes}, {num_nodes})"
            )
        return adj_tensor

    def forward(
        self, x: torch.Tensor, adj: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Thực hiện dự báo không-thời gian.

        Args:
            x: Tensor đầu vào có kích thước (Batch, Time_in, Nodes, In_features)
            adj: Ma trận kề đồ thị kích thước (Nodes, Nodes) — hỗ trợ đồ thị toàn cục hoặc đồ thị con

        Returns:
            Tensor dự báo có kích thước (Batch, Time_out, Nodes, Out_features)
        """
        b, t_in, n, f_in = x.shape
        adj_tensor = self._resolve_adjacency(adj, n, x)

        # Chuyển về định dạng Conv2d: (B, C, T, N)
        h = x.permute(0, 3, 1, 2).contiguous()

        for block in self.st_blocks:
            h = block(h, adj_tensor)

        # Output Block
        h = self.out_tconv(h)  # (B, hidden_dim, 1, N)
        h = self.out_norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        out = self.out_proj(h)  # (B, time_out * out_features, 1, N)

        # Đưa về shape chuẩn: (B, Time_out, Nodes, Out_features)
        out = out.squeeze(2).view(b, self.time_out, self.out_features, n)
        return out.permute(0, 1, 3, 2).contiguous()
