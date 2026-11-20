"""Traffic Federated Learning Benchmark CLI & Execution Engine (run.py)

Entry point to execute Centralized, Local, and Federated Learning (FedAvg / FedProx)
on Spatio-Temporal traffic forecasting datasets (METR-LA).
"""

import os
import sys
import copy
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Internal modules
from src.utils.seed import set_seed
from src.utils.metrics import (
    load_scaler,
    denormalize,
    masked_mae,
    masked_rmse,
    masked_mape,
)
from src.data.dataset import load_metr_la, StandardTrafficDataset
from src.models import (
    STGCNForecaster,
    GRUForecaster,
    LSTMForecaster,
    BaseForecaster,
)
from src.federated.client import ClientNode
from src.federated.server import FederatedServer
from src.federated.trainer import FederatedTrainer


# =====================================================================
# 1. MODEL FACTORY
# =====================================================================

def build_model(
    model_name: str,
    num_nodes: int,
    adj_matrix: Optional[Union[np.ndarray, torch.Tensor]],
    cfg: Dict[str, Any],
) -> BaseForecaster:
    """Khởi tạo mô hình theo tham số chi tiết (STGCN, GRU, LSTM).

    Args:
        model_name: Tên mô hình ('STGCN', 'GRU', 'LSTM').
        num_nodes: Số trạm cảm biến (toàn cục 207 hoặc đồ thị con N_k).
        adj_matrix: Ma trận kề đồ thị (dành cho STGCN).
        cfg: Dictionary cấu hình hệ thống.

    Returns:
        Instance của BaseForecaster (nn.Module).
    """
    dataset_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})

    in_features = int(dataset_cfg.get("features", 2))
    time_in = int(dataset_cfg.get("time_in", 12))
    time_out = int(dataset_cfg.get("time_out", 12))
    out_features = int(dataset_cfg.get("out_features", 1))

    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))

    name = model_name.upper()
    if name == "STGCN":
        kernel_size_t = int(model_cfg.get("kernel_size_t", 3))
        kernel_size_s = int(model_cfg.get("kernel_size_s", 3))
        conv_type = str(model_cfg.get("conv_type", "cheb"))

        # Đồng bộ default_adj sang torch.Tensor float32 ngay trong build_model
        adj_tensor = (
            torch.as_tensor(adj_matrix, dtype=torch.float32)
            if adj_matrix is not None
            else None
        )

        return STGCNForecaster(
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            kernel_size_t=kernel_size_t,
            kernel_size_s=kernel_size_s,
            conv_type=conv_type,
            default_adj=adj_tensor,
        )
    elif name == "GRU":
        return GRUForecaster(
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif name == "LSTM":
        return LSTMForecaster(
            num_nodes=num_nodes,
            in_features=in_features,
            time_in=time_in,
            time_out=time_out,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Mô hình '{model_name}' không được hỗ trợ. Các lựa chọn: STGCN, GRU, LSTM."
        )


# =====================================================================
# 2. EVALUATION & METRICS HELPERS
# =====================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: float,
    std: float,
    adj: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Dự báo và đánh giá mô hình bằng các metric trong không gian thực tế (mph).

    Returns:
        Dict gồm MAE, RMSE, MAPE và per_node_mae.
    """
    model.eval()
    preds = []
    targets = []

    # Chuyển adj sang torch.Tensor trên cùng device trước vòng lặp
    adj_tensor = None
    if adj is not None:
        adj_tensor = torch.as_tensor(adj, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            try:
                if adj_tensor is not None:
                    out = model(batch_x, adj=adj_tensor)
                else:
                    out = model(batch_x)
            except TypeError:
                out = model(batch_x)

            preds.append(out.detach().cpu().numpy())
            targets.append(batch_y.numpy())

    y_pred = np.concatenate(preds, axis=0)  # (B_total, Time_out, N, 1)
    y_true = np.concatenate(targets, axis=0)  # (B_total, Time_out, N, 1)

    # Khử chuẩn hóa về vận tốc thực tế (mph)
    y_pred_real = denormalize(y_pred, mean, std)
    y_true_real = denormalize(y_true, mean, std)

    mae = masked_mae(y_true_real, y_pred_real)
    rmse = masked_rmse(y_true_real, y_pred_real)
    mape = masked_mape(y_true_real, y_pred_real)

    # Tính sai số từng sensor node
    num_nodes = y_true_real.shape[2]
    per_node_mae = []
    for n in range(num_nodes):
        node_true = y_true_real[:, :, n, 0]
        node_pred = y_pred_real[:, :, n, 0]
        node_mae = masked_mae(node_true, node_pred)
        per_node_mae.append(float(node_mae) if np.isfinite(node_mae) else 0.0)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "per_node_mae": per_node_mae,
    }


# =====================================================================
# 3. BENCHMARK TRAINER ENGINE
# =====================================================================

class BenchmarkTrainer(FederatedTrainer):
    """Mở rộng FederatedTrainer với khả năng đánh giá validation theo từng round/epoch,
    lưu vết lịch sử huấn luyện (history) và theo dõi mô hình tốt nhất (best_model).
    """

    def __init__(
        self,
        model_builder,
        device: torch.device,
        config: Dict[str, Any],
    ):
        super().__init__(model_builder, device, config)
        self.history: List[Dict[str, Any]] = []
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_mae: float = float("inf")
        self.best_step: int = 0

    def train_centralized(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        eval_fn: Optional[Any] = None,
    ) -> nn.Module:
        """Huấn luyện tập trung (Centralized Learning - Upper-bound)."""
        model = self.model_builder().to(self.device)
        lr = self.config.get("lr", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        epochs = self.config.get("epochs", 10)

        print("\n--- Bắt đầu huấn luyện với Centralized Learning ---")
        self.history = []
        self.best_val_mae = float("inf")
        self.best_step = 0
        self.best_model_state = copy.deepcopy(model.state_dict())

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
            record: Dict[str, Any] = {"epoch": epoch + 1, "train_loss": epoch_loss}

            # Đánh giá trên tập validation sau mỗi epoch
            if val_loader is not None and eval_fn is not None:
                val_metrics = eval_fn(model, val_loader)
                record["val_mae"] = val_metrics["MAE"]
                record["val_rmse"] = val_metrics["RMSE"]
                record["val_mape"] = val_metrics["MAPE"]

                print(
                    f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f} | "
                    f"Val MAE: {val_metrics['MAE']:.4f} mph | "
                    f"RMSE: {val_metrics['RMSE']:.4f} | MAPE: {val_metrics['MAPE']:.2f}%"
                )

                if val_metrics["MAE"] < self.best_val_mae:
                    self.best_val_mae = val_metrics["MAE"]
                    self.best_model_state = copy.deepcopy(model.state_dict())
            else:
                print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {epoch_loss:.4f}")

            self.history.append(record)

        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

        return model

    def train_federated(
        self,
        client_loaders: List[DataLoader],
        val_loader: DataLoader,
        eval_fn: Optional[Any] = None,
    ) -> nn.Module:
        """Huấn luyện liên kết (Federated Learning - FedAvg / FedProx)."""
        rounds = self.config.get("rounds", 10)
        local_epochs = self.config.get("local_epochs", 1)
        lr = self.config.get("lr", 0.001)
        mu = self.config.get("mu", 0.0)

        global_model = self.model_builder()
        server = FederatedServer(global_model, self.device)

        clients = [
            ClientNode(
                client_id=i,
                model=self._create_client_model(i),
                train_loader=loader,
                device=self.device,
                epochs=local_epochs,
                lr=lr,
                mu=mu,
            )
            for i, loader in enumerate(client_loaders)
        ]

        algo_name = "FedProx" if mu > 0 else "FedAvg"
        print(f"\n--- Bắt đầu huấn luyện Federated Learning ({algo_name}, {rounds} rounds, local_epochs={local_epochs}, mu={mu}) ---")

        self.history = []
        self.best_val_mae = float("inf")
        self.best_model_state = copy.deepcopy(server.get_global_weights())

        for r in range(rounds):
            global_weights = server.get_global_weights()
            client_results = []

            for client in clients:
                client.set_weights(global_weights)
                weights, num_samples = client.train()
                client_results.append((weights, num_samples))

            # Tổng hợp trọng số tại Server
            server.aggregate(client_results)

            record: Dict[str, Any] = {"round": r + 1}

            # Đánh giá Global Model trên val_loader sau mỗi communication round
            if val_loader is not None and eval_fn is not None:
                val_metrics = eval_fn(server.global_model, val_loader)
                record["val_mae"] = val_metrics["MAE"]
                record["val_rmse"] = val_metrics["RMSE"]
                record["val_mape"] = val_metrics["MAPE"]

                print(
                    f"Round [{r+1:02d}/{rounds:02d}] - "
                    f"Val MAE: {val_metrics['MAE']:.4f} mph | "
                    f"RMSE: {val_metrics['RMSE']:.4f} | MAPE: {val_metrics['MAPE']:.2f}%"
                )

                if val_metrics["MAE"] < self.best_val_mae:
                    self.best_val_mae = val_metrics["MAE"]
                    self.best_model_state = copy.deepcopy(server.get_global_weights())
            else:
                print(f"Round [{r+1:02d}/{rounds:02d}] hoàn thành.")

            self.history.append(record)

        if self.best_model_state is not None:
            server.global_model.load_state_dict(self.best_model_state)

        return server.global_model

    def train_local(
        self,
        client_loaders: List[DataLoader],
        epochs: Optional[int] = None,
    ) -> List[nn.Module]:
        """Huấn luyện độc lập tại từng Client (Local Learning - Lower-bound)."""
        local_epochs = epochs if epochs is not None else self.config.get("epochs", 10)
        lr = self.config.get("lr", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-4)

        trained_models = []
        self.history = []

        print(f"\n--- Bắt đầu huấn luyện Local Learning ({len(client_loaders)} Clients, {local_epochs} Epochs) ---")
        for i, loader in enumerate(client_loaders):
            print(f"Đang huấn luyện Client độc lập [{i+1}/{len(client_loaders)}]...")
            client_model = self._create_client_model(i).to(self.device)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()

            for epoch in range(local_epochs):
                client_model.train()
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
                if (epoch + 1) == local_epochs or (epoch + 1) % max(1, local_epochs // 3) == 0:
                    print(f"  Client {i} - Epoch [{epoch+1:02d}/{local_epochs:02d}] - Loss: {epoch_loss:.4f}")

            trained_models.append(client_model)

        return trained_models


# =====================================================================
# 4. LOGGING & SUMMARY UTILITIES
# =====================================================================

def log_summary_csv(csv_path: str, row_dict: Dict[str, Any]) -> None:
    """Ghi dòng kết quả thực nghiệm vào bảng tổng hợp CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "timestamp",
        "run_name",
        "model",
        "mode",
        "scenario",
        "algo",
        "rounds",
        "epochs",
        "batch_size",
        "lr",
        "mu",
        "seed",
        "test_mae",
        "test_rmse",
        "test_mape",
        "checkpoint_path",
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in fieldnames})


# =====================================================================
# 5. MAIN EXECUTION PIPELINE
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traffic Federated Learning Benchmark CLI (run.py)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn đến file cấu hình YAML mặc định.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["STGCN", "GRU", "LSTM", "stgcn", "gru", "lstm"],
        help="Kiến trúc mô hình dự báo.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["centralized", "local", "federated"],
        help="Chế độ chạy: centralized, local, hoặc federated.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["FedAvg", "FedProx", "fedavg", "fedprox"],
        help="Thuật toán học liên kết (dành cho chế độ federated).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["spatial_skew", "temporal_shift"],
        help="Kịch bản phân chia Non-IID.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Số communication rounds trong Federated Learning.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Số epoch (hoặc local_epochs trong FL).",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Kích thước batch dữ liệu.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        help="Hệ số phạt proximal term cho FedProx.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed tái lập kết quả.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy nhanh trên tập dữ liệu siêu nhỏ để kiểm tra thông suốt luồng.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Đọc cấu hình YAML
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ghi đè cấu hình từ tham số dòng lệnh CLI (nếu có)
    if args.model:
        config["model"]["name"] = args.model.upper()
    if args.mode:
        config["training"]["mode"] = args.mode.lower()
    if args.algo:
        config["federated"]["algorithm"] = "FedProx" if args.algo.lower() == "fedprox" else "FedAvg"
    if args.scenario:
        config["non_iid"]["scenario"] = args.scenario.lower()
    if args.rounds:
        config["federated"]["rounds"] = args.rounds
    if args.epochs:
        if config["training"]["mode"] == "federated":
            config["federated"]["local_epochs"] = args.epochs
        else:
            config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.mu is not None:
        config["federated"]["mu"] = args.mu
    if args.seed is not None:
        config["project"]["seed"] = args.seed

    # Điều chỉnh tham số nếu bật --smoke-test
    if args.smoke_test:
        print("\n>>> [SMOKE-TEST] KÍCH HOẠT CHẾ ĐỘ KIỂM THỬ NHANH (TINY SLICE) <<<")
        if args.epochs is None:
            config["training"]["epochs"] = 1
            config["federated"]["local_epochs"] = 1
        if args.rounds is None:
            config["federated"]["rounds"] = 2

    # Trích xuất các tham số quan trọng
    seed = int(config.get("project", {}).get("seed", 42))
    mode = config.get("training", {}).get("mode", "centralized").lower()
    model_name = config.get("model", {}).get("name", "STGCN").upper()
    scenario = config.get("non_iid", {}).get("scenario", "spatial_skew").lower()
    algo = config.get("federated", {}).get("algorithm", "FedAvg")
    batch_size = int(config.get("training", {}).get("batch_size", 64))

    # Cố định Seed
    set_seed(seed)

    # Chọn thiết bị phần cứng
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] Sử dụng CPU")

    print("\n" + "=" * 65)
    print(" TRAFFIC FL BENCHMARK RUNNER")
    print("=" * 65)
    print(f"Model       : {model_name}")
    print(f"Mode        : {mode}")
    print(f"Scenario    : {scenario if mode != 'centralized' else 'N/A'}")
    print(f"Algorithm   : {algo if mode == 'federated' else 'N/A'}")
    print(f"Batch Size  : {batch_size}")
    print(f"Seed        : {seed}")
    print("=" * 65)

    # 2. Nạp dữ liệu gốc & Scaler
    processed_dir = config.get("dataset", {}).get("processed_dir", "data/processed/")
    dataset_name = config.get("dataset", {}).get("name", "METR-LA")
    data_path = os.path.join(processed_dir, dataset_name)

    print(f"\n[1/5] Nạp dữ liệu từ: {data_path} ...")
    data_dict = load_metr_la(data_path)

    train_x = data_dict["train_x"]
    train_y = data_dict["train_y"]
    val_x = data_dict["val_x"]
    val_y = data_dict["val_y"]
    test_x = data_dict["test_x"]
    test_y = data_dict["test_y"]
    adj_global = data_dict["adjacency"]

    scaler_path = os.path.join(data_path, "scaler.pkl")
    mean, std = load_scaler(scaler_path)
    print(f"-> Nạp Scaler thành công: mean={mean:.4f}, std={std:.4f}")

    # Áp dụng cắt lát dữ liệu cho --smoke-test
    if args.smoke_test:
        train_x = train_x[:64]
        train_y = train_y[:64]
        val_x = val_x[:32]
        val_y = val_y[:32]
        test_x = test_x[:32]
        test_y = test_y[:32]
        print(f"-> Slice smoke-test: Train={train_x.shape[0]}, Val={val_x.shape[0]}, Test={test_x.shape[0]}")

    # Tạo DataLoader cho tập Validation và Test toàn cục (207 nodes)
    val_ds = StandardTrafficDataset(val_x, val_y)
    test_ds = StandardTrafficDataset(test_x, test_y)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Chuẩn bị DataLoader và đồ thị con theo chế độ chạy
    client_loaders: List[DataLoader] = []
    client_subgraphs: Dict[int, Tuple[int, np.ndarray]] = {}
    client_node_indices_dict: Dict[int, List[int]] = {}

    print(f"\n[2/5] Chuẩn bị DataLoader theo chế độ '{mode}'...")

    if mode == "centralized":
        train_ds = StandardTrafficDataset(train_x, train_y)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    elif mode in ["local", "federated"]:
        splits_dir = Path(config.get("non_iid", {}).get("splits_dir", "splits/"))

        if scenario == "spatial_skew":
            split_file = splits_dir / "spatial_skew.json"
            if not split_file.exists():
                raise FileNotFoundError(f"Không tìm thấy file kịch bản: {split_file}")

            with open(split_file, "r", encoding="utf-8") as f:
                split_cfg = json.load(f)

            clients_info = split_cfg.get("clients", {})
            sorted_keys = sorted(
                clients_info.keys(),
                key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x),
            )

            print(f"-> Phân rã dữ liệu Spatial Skew cho {len(sorted_keys)} clients...")
            for idx, c_key in enumerate(sorted_keys):
                node_indices = clients_info[c_key]["node_indices"]
                n_k = len(node_indices)
                client_node_indices_dict[idx] = node_indices

                # Cắt tensor con cho client k
                c_train_x = train_x[:, :, node_indices, :]
                c_train_y = train_y[:, :, node_indices, :]

                # Cắt ma trận kề con A_k tương ứng
                sub_adj = adj_global[np.ix_(node_indices, node_indices)]
                client_subgraphs[idx] = (n_k, sub_adj)

                # Bọc TensorDataset để hỗ trợ kích thước nút con N_k != 207
                c_ds = TensorDataset(
                    torch.from_numpy(np.asarray(c_train_x, dtype=np.float32)),
                    torch.from_numpy(np.asarray(c_train_y, dtype=np.float32)),
                )
                c_loader = DataLoader(c_ds, batch_size=batch_size, shuffle=True)
                client_loaders.append(c_loader)
                print(f"   [Client {idx}] Nodes: {n_k} | Sub-Adj: {sub_adj.shape} | Samples: {len(c_ds)}")

        elif scenario == "temporal_shift":
            split_file = splits_dir / "temporal_shift.json"
            if not split_file.exists():
                raise FileNotFoundError(f"Không tìm thấy file kịch bản: {split_file}")

            with open(split_file, "r", encoding="utf-8") as f:
                split_cfg = json.load(f)

            clients_info = split_cfg.get("clients", {})
            sorted_keys = sorted(
                clients_info.keys(),
                key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x),
            )

            print(f"-> Phân rã dữ liệu Temporal Shift cho {len(sorted_keys)} clients...")
            for idx, c_key in enumerate(sorted_keys):
                sample_indices = clients_info[c_key]["sample_indices"]

                # Lọc index nếu chạy smoke-test
                if args.smoke_test:
                    valid_idx = [i for i in sample_indices if i < len(train_x)]
                    if not valid_idx:
                        valid_idx = list(range(idx, len(train_x), len(sorted_keys)))
                    sample_indices = valid_idx

                c_train_x = train_x[sample_indices]
                c_train_y = train_y[sample_indices]

                # Temporal shift giữ nguyên toàn bộ 207 nodes nên dùng được StandardTrafficDataset
                c_ds = StandardTrafficDataset(c_train_x, c_train_y)
                c_loader = DataLoader(c_ds, batch_size=batch_size, shuffle=True)
                client_loaders.append(c_loader)
                print(f"   [Client {idx}] Nodes: 207 | Samples: {len(c_ds)}")

        else:
            raise ValueError(f"Kịch bản Non-IID '{scenario}' không hợp lệ.")

    # 4. Khởi tạo Model Builder & Trainer Engine
    print("\n[3/5] Khởi tạo Model Builder & FederatedTrainer...")

    def model_builder(client_id: Optional[int] = None) -> BaseForecaster:
        """Tạo instance mô hình tương thích với client_id và cấu hình toàn cục."""
        if (
            client_id is not None
            and scenario == "spatial_skew"
            and client_id in client_subgraphs
        ):
            n_nodes, sub_adj = client_subgraphs[client_id]
            return build_model(model_name, n_nodes, sub_adj, config)

        # Mặc định trả về mô hình toàn cục 207 nút
        return build_model(model_name, 207, adj_global, config)

    # Xác định giá trị mu cho thuật toán
    if algo.lower() == "fedavg":
        trainer_mu = 0.0
    else:
        trainer_mu = float(config.get("federated", {}).get("mu", 0.01))

    trainer_cfg = {
        "lr": float(config.get("training", {}).get("learning_rate", 0.001)),
        "weight_decay": float(config.get("training", {}).get("weight_decay", 0.0001)),
        "epochs": int(config.get("training", {}).get("epochs", 10)),
        "rounds": int(config.get("federated", {}).get("rounds", 10)),
        "local_epochs": int(config.get("federated", {}).get("local_epochs", 1)),
        "mu": trainer_mu,
    }

    trainer = BenchmarkTrainer(
        model_builder=model_builder,
        device=device,
        config=trainer_cfg,
    )

    eval_fn = lambda m, l: evaluate_model(m, l, device, mean, std, adj=adj_global)

    # 5. Thực thi Huấn luyện
    print(f"\n[4/5] Khởi chạy huấn luyện theo chế độ '{mode}'...")
    trained_model = None
    local_models = None

    if mode == "centralized":
        trained_model = trainer.train_centralized(
            train_loader=train_loader,
            val_loader=val_loader,
            eval_fn=eval_fn,
        )

    elif mode == "federated":
        trained_model = trainer.train_federated(
            client_loaders=client_loaders,
            val_loader=val_loader,
            eval_fn=eval_fn,
        )

    elif mode == "local":
        # Chế độ local (Lower-bound) sử dụng training.epochs để đảm bảo công bằng số lượng cập nhật với centralized/federated
        local_epochs_target = config.get("training", {}).get("epochs", 10)
        local_models = trainer.train_local(
            client_loaders=client_loaders,
            epochs=local_epochs_target,
        )

    # 6. Đánh giá cuối cùng trên Test Loader
    print("\n[5/5] Đánh giá trên tập kiểm thử (Test Evaluation)...")
    test_metrics: Dict[str, Any] = {}

    if mode in ["centralized", "federated"]:
        assert trained_model is not None
        test_metrics = evaluate_model(
            model=trained_model,
            loader=test_loader,
            device=device,
            mean=mean,
            std=std,
            adj=adj_global,
        )
        print(f"\n>>> KẾT QUẢ TEST ({mode.upper()}) <<<")
        print(f"MAE  : {test_metrics['MAE']:.4f} mph")
        print(f"RMSE : {test_metrics['RMSE']:.4f} mph")
        print(f"MAPE : {test_metrics['MAPE']:.2f}%")

    elif mode == "local":
        assert local_models is not None
        print(f"Đang đánh giá {len(local_models)} Client Models trên Test Set...")
        client_test_maes = []
        client_test_rmses = []
        client_test_mapes = []
        full_per_node_mae = [0.0] * 207

        if scenario == "spatial_skew":
            for idx, c_model in enumerate(local_models):
                node_indices = client_node_indices_dict[idx]
                _, sub_adj = client_subgraphs[idx]

                c_test_x = test_x[:, :, node_indices, :]
                c_test_y = test_y[:, :, node_indices, :]

                c_test_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(np.asarray(c_test_x, dtype=np.float32)),
                        torch.from_numpy(np.asarray(c_test_y, dtype=np.float32)),
                    ),
                    batch_size=batch_size,
                    shuffle=False,
                )

                c_metrics = evaluate_model(
                    model=c_model,
                    loader=c_test_loader,
                    device=device,
                    mean=mean,
                    std=std,
                    adj=sub_adj,
                )
                client_test_maes.append(c_metrics["MAE"])
                client_test_rmses.append(c_metrics["RMSE"])
                client_test_mapes.append(c_metrics["MAPE"])

                # Điền per-node MAE
                for local_i, global_i in enumerate(node_indices):
                    full_per_node_mae[global_i] = c_metrics["per_node_mae"][local_i]

            test_metrics = {
                "MAE": float(np.mean(client_test_maes)),
                "RMSE": float(np.mean(client_test_rmses)),
                "MAPE": float(np.mean(client_test_mapes)),
                "per_node_mae": full_per_node_mae,
            }

        else:  # temporal_shift
            client_per_node_maes = []
            for c_model in local_models:
                c_metrics = evaluate_model(
                    model=c_model,
                    loader=test_loader,
                    device=device,
                    mean=mean,
                    std=std,
                    adj=adj_global,
                )
                client_test_maes.append(c_metrics["MAE"])
                client_test_rmses.append(c_metrics["RMSE"])
                client_test_mapes.append(c_metrics["MAPE"])
                client_per_node_maes.append(c_metrics["per_node_mae"])

            # Trung bình cộng per_node_mae của toàn bộ các client độc lập
            avg_per_node_mae = np.mean(client_per_node_maes, axis=0).tolist()

            test_metrics = {
                "MAE": float(np.mean(client_test_maes)),
                "RMSE": float(np.mean(client_test_rmses)),
                "MAPE": float(np.mean(client_test_mapes)),
                "per_node_mae": avg_per_node_mae,
            }

        print(f"\n>>> KẾT QUẢ TEST TRUNG BÌNH (LOCAL BENCHMARK) <<<")
        print(f"Mean MAE  : {test_metrics['MAE']:.4f} mph")
        print(f"Mean RMSE : {test_metrics['RMSE']:.4f} mph")
        print(f"Mean MAPE : {test_metrics['MAPE']:.2f}%")

    # 7. Lưu trữ Checkpoint & Logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo_tag = algo.lower() if mode == "federated" else "none"
    scenario_tag = scenario if mode != "centralized" else "central"
    run_name = f"{model_name.lower()}_{mode}_{scenario_tag}_{algo_tag}_seed{seed}_{timestamp}"

    # Checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / f"{run_name}.pt")

    if trained_model is not None:
        torch.save(trained_model.state_dict(), checkpoint_path)
    elif local_models is not None:
        torch.save(
            {"local_models": [m.state_dict() for m in local_models]},
            checkpoint_path,
        )

    print(f"\n[Artifacts] Đã lưu checkpoint tại: {checkpoint_path}")

    # Logs JSON
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_json_path = logs_dir / f"{run_name}.json"

    experiment_log = {
        "run_name": run_name,
        "timestamp": timestamp,
        "config": config,
        "history": trainer.history,
        "test_metrics": {
            "MAE": test_metrics.get("MAE"),
            "RMSE": test_metrics.get("RMSE"),
            "MAPE": test_metrics.get("MAPE"),
        },
        "per_node_mae": test_metrics.get("per_node_mae", []),
        "checkpoint_path": checkpoint_path,
    }

    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=4)
    print(f"[Artifacts] Đã lưu log chi tiết tại: {log_json_path}")

    # Summary CSV
    summary_csv_path = "logs/summary.csv"
    summary_row = {
        "timestamp": timestamp,
        "run_name": run_name,
        "model": model_name,
        "mode": mode,
        "scenario": scenario if mode != "centralized" else "none",
        "algo": algo if mode == "federated" else "none",
        "rounds": config.get("federated", {}).get("rounds", 0) if mode == "federated" else 0,
        "epochs": (
            config.get("federated", {}).get("local_epochs", 0)
            if mode == "federated"
            else config.get("training", {}).get("epochs", 0)
        ),
        "batch_size": batch_size,
        "lr": config.get("training", {}).get("learning_rate", 0.001),
        "mu": trainer_mu if mode == "federated" else 0.0,
        "seed": seed,
        "test_mae": f"{test_metrics.get('MAE', 0.0):.4f}",
        "test_rmse": f"{test_metrics.get('RMSE', 0.0):.4f}",
        "test_mape": f"{test_metrics.get('MAPE', 0.0):.2f}",
        "checkpoint_path": checkpoint_path,
    }
    log_summary_csv(summary_csv_path, summary_row)
    print(f"[Artifacts] Đã tổng hợp kết quả vào: {summary_csv_path}")

    print("\n" + "=" * 65)
    print(" HUẤN LUYỆN VÀ ĐÁNH GIÁ THÀNH CÔNG!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
