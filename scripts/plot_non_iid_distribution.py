import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import load_metr_la


def load_scaler(scaler_path: str = "data/processed/METR-LA/scaler.pkl"):
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def denormalize_speed(speed_norm: np.ndarray, scaler) -> np.ndarray:
    if isinstance(scaler, dict):
        mean = scaler.get("mean", scaler.get("mu", 0.0))
        std = scaler.get("std", scaler.get("sigma", 1.0))
        return speed_norm * std + mean
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return speed_norm * scaler.scale_[0] + scaler.mean_[0]
    if hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(speed_norm.reshape(-1, 1)).reshape(speed_norm.shape)
    return speed_norm


def main():
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_metr_la("data/processed/METR-LA")
    X_train = data["train_x"]
    scaler = load_scaler("data/processed/METR-LA/scaler.pkl")

    # Vận tốc thực tế (mph) tại timestep hiện tại (t=11, feature 0)
    speed_mph = denormalize_speed(X_train[:, -1, :, 0], scaler)

    # 1. Vẽ phân phối Spatial Skew (phân bố vận tốc trung bình của các node thuộc từng Client)
    with open("splits/spatial_skew.json", "r", encoding="utf-8") as f:
        spatial_cfg = json.load(f)["clients"]

    plt.figure(figsize=(10, 5))
    spatial_data = []
    spatial_labels = []
    for cid, info in spatial_cfg.items():
        node_idx = info["node_indices"]
        client_speeds = speed_mph[:, node_idx].reshape(-1)
        spatial_data.append(client_speeds)
        spatial_labels.append(f"{cid}\n({len(node_idx)} nodes)")

    plt.boxplot(spatial_data, labels=spatial_labels, showfliers=False)
    plt.title("METR-LA Non-IID Scenario 1: Spatial Skew (Speed Distribution per Client)")
    plt.ylabel("Speed (mph)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_skew_dist.png", dpi=200)
    plt.close()

    # 2. Vẽ phân phối Temporal Shift (tỷ lệ mẫu giờ cao điểm vs thấp điểm giữa các Client)
    with open("splits/temporal_shift.json", "r", encoding="utf-8") as f:
        temporal_cfg = json.load(f)["clients"]

    client_names = list(temporal_cfg.keys())
    peak_counts = [temporal_cfg[c]["peak_samples"] for c in client_names]
    offpeak_counts = [temporal_cfg[c]["offpeak_samples"] for c in client_names]

    x_pos = np.arange(len(client_names))
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos, peak_counts, label="Peak Hours (07-10h, 16-19h)", color="#d9534f")
    plt.bar(x_pos, offpeak_counts, bottom=peak_counts, label="Off-Peak Hours", color="#5bc0de")
    plt.xticks(x_pos, client_names)
    plt.title("METR-LA Non-IID Scenario 2: Temporal Shift (Peak vs Off-Peak Samples)")
    plt.ylabel("Number of Training Samples")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "temporal_shift_dist.png", dpi=200)
    plt.close()

    print(f"Đã xuất biểu đồ Non-IID thành công tại: {output_dir.resolve()}")


if __name__ == "__main__":
    main()