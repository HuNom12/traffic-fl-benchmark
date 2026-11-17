from typing import Dict, Any, Optional, Union
import numpy as np
import torch

from src.models.base import BaseForecaster
from src.models.temporal import GRUForecaster, LSTMForecaster
from src.models.spatio_temporal import STGCNForecaster

# Aliases tiện lợi khớp với tên gọi trong config.yaml
GRU = GRUForecaster
LSTM = LSTMForecaster
STGCN = STGCNForecaster

MODEL_REGISTRY = {
    "GRU": GRUForecaster,
    "LSTM": LSTMForecaster,
    "STGCN": STGCNForecaster,
}


def build_model(
    config: Dict[str, Any],
    default_adj: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> BaseForecaster:
    """Khởi tạo mô hình dự báo từ dictionary cấu hình YAML."""
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    model_name = model_cfg.get("name", "STGCN").upper()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Mô hình '{model_name}' không được hỗ trợ. Các lựa chọn: {list(MODEL_REGISTRY.keys())}"
        )

    num_nodes = int(dataset_cfg.get("num_nodes", 207))
    in_features = int(dataset_cfg.get("features", 2))
    time_in = int(dataset_cfg.get("time_in", 12))
    time_out = int(dataset_cfg.get("time_out", 12))
    out_features = int(dataset_cfg.get("out_features", 1))

    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))

    common_kwargs = {
        "num_nodes": num_nodes,
        "in_features": in_features,
        "time_in": time_in,
        "time_out": time_out,
        "out_features": out_features,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    if model_name == "STGCN":
        common_kwargs["kernel_size_t"] = int(model_cfg.get("kernel_size_t", 3))
        common_kwargs["kernel_size_s"] = int(model_cfg.get("kernel_size_s", 3))
        common_kwargs["conv_type"] = str(model_cfg.get("conv_type", "cheb"))
        if default_adj is not None:
            common_kwargs["default_adj"] = default_adj

    return MODEL_REGISTRY[model_name](**common_kwargs)


__all__ = [
    "BaseForecaster",
    "GRUForecaster",
    "LSTMForecaster",
    "STGCNForecaster",
    "GRU",
    "LSTM",
    "STGCN",
    "build_model",
]

