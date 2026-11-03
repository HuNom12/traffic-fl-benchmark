import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import load_and_preprocess_pems


def plot_distributions():
    print("=== [Visualizing Non-IID Partition Distributions] ===")
    os.makedirs("reports", exist_ok=True)

    # 1. Nạp dữ liệu giao thông gốc
    data_dict = load_and_preprocess_pems("data/raw/pems04.npz", time_in=12, time_out=12)
    train_y = data_dict["train_y"]  # (Samples, 12, Nodes, 1)

    # Đưa về giá trị lưu lượng thực tế nếu có mean/std để đồ thị trực quan hơn
    mean = data_dict.get("mean", 0.0)
    std = data_dict.get("std", 1.0)
    raw_flow_y = (train_y * std) + mean

    # ---------------------------------------------------------
    # 2. Trực quan hóa Kịch bản 1: Spatial Skew
    # ---------------------------------------------------------
    spatial_file = "splits/spatial_skew.json"
    if os.path.exists(spatial_file):
        with open(spatial_file, "r", encoding="utf-8") as f:
            spatial_cfg = json.load(f)

        client_data_spatial = []
        labels_spatial = []

        for c_id, client_info in spatial_cfg["clients"].items():
            nodes = client_info["sensor_nodes"]
            if len(nodes) > 0:
                # Lấy toàn bộ lưu lượng dự báo thuộc tập sensor của Client này
                client_flows = raw_flow_y[:, :, nodes, 0].flatten()
                client_data_spatial.append(client_flows)
                labels_spatial.append(f"Client {c_id}\n({len(nodes)} nodes)")

        plt.figure(figsize=(10, 6))
        plt.boxplot(client_data_spatial, tick_labels=labels_spatial, patch_artist=True, showfliers=False)
        plt.title("Scenario 1: Spatial Skew (Traffic Flow Distribution per Client)", fontsize=14, pad=15)
        plt.ylabel("Traffic Flow (Vehicles / 5-min)", fontsize=12)
        plt.xlabel("Federated Clients", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        save_spatial_path = "reports/spatial_skew_distribution.png"
        plt.tight_layout()
        plt.savefig(save_spatial_path, dpi=300)
        plt.close()
        print(f"-> Đã xuất đồ thị Spatial Skew tại: {save_spatial_path}")
    else:
        print(f"-> Không tìm thấy {spatial_file}, bỏ qua vẽ Spatial Skew.")

    # ---------------------------------------------------------
    # 3. Trực quan hóa Kịch bản 2: Temporal Shift
    # ---------------------------------------------------------
    temporal_file = "splits/temporal_shift.json"
    if os.path.exists(temporal_file):
        with open(temporal_file, "r", encoding="utf-8") as f:
            temporal_cfg = json.load(f)

        client_data_temporal = []
        labels_temporal = []

        for c_id, client_info in temporal_cfg["clients"].items():
            sample_indices = client_info["sample_indices"]
            if len(sample_indices) > 0:
                # Lấy lưu lượng trên tất cả các node trong khoảng thời gian của Client
                client_flows = raw_flow_y[sample_indices, :, :, 0].flatten()
                client_data_temporal.append(client_flows)
                labels_temporal.append(f"Client {c_id}\n({len(sample_indices)} timesteps)")

        plt.figure(figsize=(10, 6))
        plt.boxplot(client_data_temporal, tick_labels=labels_temporal, patch_artist=True, showfliers=False)
        plt.title("Scenario 2: Temporal Shift (Traffic Flow Distribution per Time Window)", fontsize=14, pad=15)
        plt.ylabel("Traffic Flow (Vehicles / 5-min)", fontsize=12)
        plt.xlabel("Federated Clients", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        save_temporal_path = "reports/temporal_shift_distribution.png"
        plt.tight_layout()
        plt.savefig(save_temporal_path, dpi=300)
        plt.close()
        print(f"-> Đã xuất đồ thị Temporal Shift tại: {save_temporal_path}")
    else:
        print(f"-> Không tìm thấy {temporal_file}, bỏ qua vẽ Temporal Shift.")



if __name__ == "__main__":
    plot_distributions()