import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.metrics import (
    denormalize,
    _get_valid_mask,
    masked_mae,
    masked_rmse,
    masked_mape,
)


# =====================================================================
# 1. TÍNH TOÁN SAI SỐ TỪNG NÚT & TÍNH CÔNG BẰNG (FAIRNESS & WORST-DECILE)
# =====================================================================

def compute_per_node_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_idx: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Tính toán các chỉ số sai số (MAE, RMSE, MAPE) riêng lẻ cho từng cảm biến (Per-Node Error).

    Args:
        y_true: Mảng thực tế có shape (Num_samples, Time_out, Num_nodes) hoặc (Num_samples, Num_nodes) theo mph.
        y_pred: Mảng dự báo có shape tương ứng theo mph.
        horizon_idx: Chỉ số bước dự báo cần đánh giá (0-based, vd: 2 cho 15 phút, 5 cho 30 phút, 11 cho 60 phút).
                     Nếu None, tính trung bình qua tất cả các bước thời gian.

    Returns:
        Dict chứa:
            - 'mae': Mảng 1D độ lớn N chứa MAE của từng nút.
            - 'rmse': Mảng 1D độ lớn N chứa RMSE của từng nút.
            - 'mape': Mảng 1D độ lớn N chứa MAPE của từng nút.
    """
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    if y_true.ndim == 3:
        if horizon_idx is not None:
            # Chọn riêng bước dự báo được chỉ định
            yt = y_true[:, horizon_idx, :]  # (Samples, Nodes)
            yp = y_pred[:, horizon_idx, :]  # (Samples, Nodes)
        else:
            # Gộp chiều Samples và Time: (Samples * Time, Nodes)
            s, t, n = y_true.shape
            yt = y_true.reshape(-1, n)
            yp = y_pred.reshape(-1, n)
    elif y_true.ndim == 2:
        yt = y_true
        yp = y_pred
    else:
        raise ValueError(f"Shape dữ liệu {y_true.shape} không hợp lệ (kỳ vọng 2D hoặc 3D)")

    num_nodes = yt.shape[1]
    node_mae = np.full(num_nodes, np.nan, dtype=np.float64)
    node_rmse = np.full(num_nodes, np.nan, dtype=np.float64)
    node_mape = np.full(num_nodes, np.nan, dtype=np.float64)

    for v in range(num_nodes):
        target_v = yt[:, v]
        pred_v = yp[:, v]
        mask = _get_valid_mask(target_v, pred_v)

        if np.any(mask):
            err = np.abs(target_v[mask] - pred_v[mask])
            node_mae[v] = np.mean(err)
            node_rmse[v] = np.sqrt(np.mean((target_v[mask] - pred_v[mask]) ** 2))
            node_mape[v] = np.mean(err / target_v[mask]) * 100.0

    return {
        "mae": node_mae,
        "rmse": node_rmse,
        "mape": node_mape,
    }


def compute_fairness_metrics(
    node_mae: np.ndarray,
    decile_ratio: float = 0.10,
) -> Dict[str, float]:
    """Phân tích tính công bằng (Fairness Analysis):
    Tính toán Worst-Node Error, Worst-Decile Error (10% nút tệ nhất) và Khoảng cách công bằng (Fairness Gap).

    Args:
        node_mae: Mảng 1D chứa MAE của N nút cảm biến.
        decile_ratio: Tỷ lệ phân vị biên (mặc định 0.10 tức 10% số nút có sai số lớn nhất).

    Returns:
        Dict chứa các chỉ số định lượng tính công bằng.
    """
    valid_mae = node_mae[np.isfinite(node_mae)]
    if len(valid_mae) == 0:
        return {
            "mean_mae": float("nan"),
            "worst_node_mae": float("nan"),
            "worst_decile_mae": float("nan"),
            "best_decile_mae": float("nan"),
            "fairness_gap": float("nan"),
        }

    sorted_mae = np.sort(valid_mae)[::-1]  # Sắp xếp giảm dần từ lỗi cao nhất đến thấp nhất
    k = max(1, int(np.ceil(len(sorted_mae) * decile_ratio)))

    worst_node = float(sorted_mae[0])
    worst_decile = float(np.mean(sorted_mae[:k]))
    best_decile = float(np.mean(sorted_mae[-k:]))
    fairness_gap = float(worst_decile / (best_decile + 1e-6))

    return {
        "mean_mae": float(np.mean(valid_mae)),
        "worst_node_mae": round(worst_node, 4),
        "worst_decile_mae": round(worst_decile, 4),
        "best_decile_mae": round(best_decile, 4),
        "fairness_gap": round(fairness_gap, 4),
    }


# =====================================================================
# 2. BẢNG TỔNG HỢP HIỆU NĂNG TOÀN MẠNG (BENCHMARK SUMMARY TABLE)
# =====================================================================

def evaluate_experiment_horizons(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Dict[str, int] = None,
) -> List[Dict[str, Any]]:
    """Đánh giá chi tiết hiệu năng theo từng mốc chân trời dự báo: 15p, 30p, 60p và Overall."""
    if horizons is None:
        horizons = {
            "15 min": 2,   # Bước 3 (0-indexed: 2)
            "30 min": 5,   # Bước 6 (0-indexed: 5)
            "60 min": 11,  # Bước 12 (0-indexed: 11)
            "Overall": None,
        }

    results = []
    for h_name, h_idx in horizons.items():
        if h_idx is not None and (y_true.ndim < 3 or h_idx >= y_true.shape[1]):
            continue

        if h_idx is not None:
            yt = y_true[:, h_idx, :]
            yp = y_pred[:, h_idx, :]
        else:
            yt = y_true
            yp = y_pred

        mask = _get_valid_mask(yt, yp)
        if np.any(mask):
            overall_mae = float(np.mean(np.abs(yt[mask] - yp[mask])))
            overall_rmse = float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))
            overall_mape = float(np.mean(np.abs(yt[mask] - yp[mask]) / yt[mask]) * 100.0)
        else:
            overall_mae = overall_rmse = overall_mape = float("nan")

        # Đánh giá per-node
        node_stats = compute_per_node_errors(y_true, y_pred, horizon_idx=h_idx)
        fairness = compute_fairness_metrics(node_stats["mae"])

        results.append({
            "horizon": h_name,
            "mae": round(overall_mae, 4),
            "rmse": round(overall_rmse, 4),
            "mape": round(overall_mape, 2),
            "worst_node_mae": fairness["worst_node_mae"],
            "worst_decile_mae": fairness["worst_decile_mae"],
            "fairness_gap": fairness["fairness_gap"],
        })

    return results


def build_benchmark_summary_dataframe(
    experiments_data: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Xây dựng DataFrame bảng so sánh chuẩn hóa giữa các mô hình và phương pháp."""
    rows = []
    for exp in experiments_data:
        meta = exp.get("metadata", {})
        horizon_results = exp.get("horizon_results", [])
        for hr in horizon_results:
            rows.append({
                "Model": meta.get("model", "STGCN"),
                "Mode": meta.get("mode", "federated"),
                "Algorithm": meta.get("algorithm", "FedAvg"),
                "Scenario": meta.get("scenario", "spatial_skew"),
                "Horizon": hr["horizon"],
                "MAE (mph)": hr["mae"],
                "RMSE (mph)": hr["rmse"],
                "MAPE (%)": hr["mape"],
                "Worst-Node MAE": hr["worst_node_mae"],
                "Worst-Decile MAE": hr["worst_decile_mae"],
                "Fairness Gap": hr["fairness_gap"],
            })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Chuyển đổi DataFrame sang bảng Markdown thuần túy không cần thư viện tabulate."""
    headers = [str(c) for c in df.columns]
    col_widths = [
        max(len(h), max((len(str(val)) for val in df[c]), default=0))
        for h, c in zip(headers, df.columns)
    ]

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "| " + " | ".join("-" * max(3, w) for w in col_widths) + " |"

    data_lines = []
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(df.columns, col_widths)) + " |"
        data_lines.append(line)

    return "\n".join([header_line, sep_line] + data_lines)


def export_benchmark_reports(
    table_df: pd.DataFrame,
    report_dir: str = "reports"
) -> Tuple[str, str]:
    """Xuất bảng tổng kết hiệu năng ra tệp CSV và bảng Markdown trong thư mục reports/."""
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "benchmark_summary.csv")
    md_path = os.path.join(report_dir, "benchmark_summary.md")

    table_df.to_csv(csv_path, index=False)

    md_content = "# Bảng So Sánh Hiệu Năng Toàn Mạng (Traffic FL Benchmark)\n\n"
    md_content += "> Đơn vị đo: Vận tốc xe tính theo dặm/giờ (mph), MAPE tính theo phần trăm (%).\n\n"
    md_content += _df_to_markdown(table_df)
    md_content += "\n\n*Worst-Decile MAE: Sai số trung bình tại 10% nút có lỗi dự báo lớn nhất.*\n"
    md_content += "*Fairness Gap: Tỷ số Worst-Decile MAE / Best-Decile MAE (giá trị càng nhỏ mô hình càng công bằng).*\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"-> Đã lưu bảng tóm tắt hiệu năng tại:\n   - {csv_path}\n   - {md_path}")
    return csv_path, md_path


# =====================================================================
# 3. TRỰC QUAN HÓA ĐỒ THỊ KHOA HỌC (PLOTTING)
# =====================================================================

def plot_comparative_convergence(
    convergence_histories: Dict[str, Dict[str, List[float]]],
    save_path: str = "reports/convergence_comparison.png",
    metric_key: str = "val_losses",
    ylabel: str = "Validation Loss (MSE)",
    title: str = "Đường Cong Hội Tụ So Sánh Giữa Các Chế Độ Học",
) -> None:
    """Vẽ biểu đồ đường cong hội tụ so sánh (Comparative Convergence Curves).

    Args:
        convergence_histories: Dict ánh xạ tên phương pháp -> dict các mảng tiến trình huấn luyện.
                               Ví dụ: {"Centralized": {"val_losses": [...]}, "FedAvg": {...}, ...}
        save_path: Đường dẫn tệp ảnh xuất ra.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e", "#e6ab02"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-."]

    for idx, (method_name, hist) in enumerate(convergence_histories.items()):
        values = hist.get(metric_key, [])
        if len(values) == 0:
            continue
        rounds = hist.get("rounds", list(range(1, len(values) + 1)))
        c = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]

        plt.plot(
            rounds,
            values,
            label=method_name,
            color=c,
            linestyle=ls,
            linewidth=2.0,
            marker="o" if len(rounds) <= 30 else None,
            markersize=4,
            alpha=0.9,
        )

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Communication Round / Epoch", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ hội tụ tại: {save_path}")


def plot_time_series_comparison(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    sensor_indices: List[int],
    sensor_labels: Optional[List[str]] = None,
    time_window_slice: slice = slice(0, 288),  # 288 bước = 24 giờ
    save_path: str = "reports/prediction_vs_ground_truth.png",
    horizon_step: int = 2,  # Bước 3 (15 phút)
) -> None:
    """Vẽ biểu đồ chuỗi thời gian dự báo (Prediction vs. Ground Truth) tại các nút tiêu biểu.

    Args:
        y_true: Mảng thực tế (Samples, Time, Nodes) theo mph.
        predictions_dict: Dict {method_name: y_pred_array} theo mph.
        sensor_indices: Danh sách các sensor id cần vẽ (ví dụ: [trung tâm, ngoại ô, worst-decile]).
        sensor_labels: Nhãn tương ứng cho từng nút (tùy chọn).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_plots = len(sensor_indices)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3.5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    method_styles = {
        "Ground Truth": {"color": "#111111", "linestyle": "-", "linewidth": 2.2},
        "Centralized":  {"color": "#7570b3", "linestyle": ":", "linewidth": 1.8},
        "Local":        {"color": "#e7298a", "linestyle": "--", "linewidth": 1.6},
        "FedAvg":       {"color": "#1b9e77", "linestyle": "-.", "linewidth": 1.8},
        "FedProx":      {"color": "#d95f02", "linestyle": "-", "linewidth": 1.8},
    }

    t_axis = np.arange(time_window_slice.stop - time_window_slice.start) * 5.0 / 60.0  # Chuyển sang Giờ (0h - 24h)

    for ax_idx, node_id in enumerate(sensor_indices):
        ax = axes[ax_idx]
        lbl = sensor_labels[ax_idx] if sensor_labels else f"Sensor #{node_id}"

        # Vẽ Ground Truth
        gt_series = y_true[time_window_slice, horizon_step, node_id].squeeze()
        ax.plot(
            t_axis,
            gt_series,
            label="Thực tế (Ground Truth)",
            **method_styles.get("Ground Truth")
        )

        # Vẽ các phương pháp dự báo
        for method_name, y_pred in predictions_dict.items():
            pred_series = y_pred[time_window_slice, horizon_step, node_id].squeeze()
            style = method_styles.get(method_name, {"linewidth": 1.5})
            ax.plot(t_axis, pred_series, label=method_name, **style, alpha=0.9)

        ax.set_title(f"Dự Báo Vận Tốc 15 Phút Tới Tại {lbl}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Vận tốc (mph)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        if ax_idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=len(predictions_dict) + 1)

    axes[-1].set_xlabel("Thời gian trong ngày (Giờ: 0h - 24h)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ chuỗi thời gian tại: {save_path}")


def plot_fairness_error_distribution(
    node_mae_dict: Dict[str, np.ndarray],
    save_path: str = "reports/fairness_error_distribution.png",
) -> None:
    """Vẽ biểu đồ hộp (Boxplot) thể hiện phân bố sai số MAE của 207 cảm biến.
    Minh họa trực quan việc FedProx kéo giảm Worst-Decile Error và thu hẹp Fairness Gap.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    data_to_plot = []
    labels = []
    worst_deciles = []

    for name, mae_arr in node_mae_dict.items():
        valid_mae = mae_arr[np.isfinite(mae_arr)]
        data_to_plot.append(valid_mae)
        labels.append(name)
        # Tính ngưỡng phân vị 90%
        p90 = np.percentile(valid_mae, 90)
        worst_deciles.append(p90)

    bp = plt.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": "#111111", "linewidth": 1.5},
        meanprops={"color": "#b2182b", "linewidth": 1.8, "linestyle": "--"},
    )

    palette = ["#bcbddc", "#fdae6b", "#a1d99b", "#9ecae1"]
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.8)

    # Đánh dấu ngưỡng 10% nút tệ nhất (90th percentile)
    for idx, p90_val in enumerate(worst_deciles, start=1):
        plt.hlines(
            p90_val,
            xmin=idx - 0.25,
            xmax=idx + 0.25,
            colors="#b2182b",
            linestyles=":",
            linewidth=2.0,
            label="Ngưỡng Worst-Decile (P90)" if idx == 1 else None,
        )

    plt.title("Phân Bố Sai Số Per-Node MAE (Đánh Giá Tính Công Bằng - Fairness)", fontsize=12, fontweight="bold")
    plt.ylabel("Sai số MAE (mph)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố sai số Fairness tại: {save_path}")


# =====================================================================
# 4. TRÌNH TẠO DỮ LIỆU THỬ NGHIỆM MẪU (MOCK LOG GENERATOR)
# =====================================================================

def generate_mock_benchmark_artifacts(
    log_dir: str = "logs",
    report_dir: str = "reports",
    num_nodes: int = 207,
    num_samples: int = 288,
    time_out: int = 12,
    mean_speed: float = 54.4,
    std_speed: float = 19.5,
) -> None:
    """Tạo bộ log thử nghiệm mẫu (Mockup Logs) phản ánh đúng tính chất giao thông METR-LA
    để chạy nghiệm thu toàn bộ pipeline Task 3.3 ngay lập tức khi chưa có đủ log huấn luyện thật.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    np.random.seed(42)
    t = np.arange(num_samples)
    # Mô phỏng chu kỳ nhịp sinh hoạt ngày đêm (cao điểm sáng/chiều giảm tốc độ)
    day_cycle = np.sin(2 * np.pi * (t % 288) / 288 - np.pi / 2)
    base_speed = mean_speed + day_cycle * 12.0  # (288,)

    # Mảng Ground Truth thực tế cho 207 node
    node_variations = np.random.uniform(-8.0, 8.0, size=(1, 1, num_nodes))
    y_true_real = np.zeros((num_samples, time_out, num_nodes), dtype=np.float32)
    for s in range(num_samples):
        for h in range(time_out):
            noise = np.random.normal(0, 2.5, size=(num_nodes,))
            y_true_real[s, h, :] = np.clip(base_speed[s] + node_variations[0, 0, :] + noise, 5.0, 75.0)

    # 4 cấu hình phương pháp mô phỏng
    method_configs = [
        {
            "id": "centralized_stgcn",
            "model": "STGCN",
            "mode": "centralized",
            "algorithm": "Adam",
            "scenario": "global",
            "err_scale": 2.8,
            "worst_tail_boost": 1.2,
        },
        {
            "id": "local_stgcn_spatial",
            "model": "STGCN",
            "mode": "local",
            "algorithm": "Adam",
            "scenario": "spatial_skew",
            "err_scale": 4.6,
            "worst_tail_boost": 2.5,  # Local bị sai số cao ở các nút cô lập
        },
        {
            "id": "fedavg_stgcn_spatial",
            "model": "STGCN",
            "mode": "federated",
            "algorithm": "FedAvg",
            "scenario": "spatial_skew",
            "err_scale": 3.4,
            "worst_tail_boost": 1.9,
        },
        {
            "id": "fedprox_stgcn_spatial",
            "model": "STGCN",
            "mode": "federated",
            "algorithm": "FedProx",
            "scenario": "spatial_skew",
            "err_scale": 3.1,
            "worst_tail_boost": 1.4,  # FedProx kéo giảm đuôi sai số tốt hơn
        },
    ]

    all_experiments = []
    convergence_histories = {}
    predictions_dict = {}
    node_mae_dict = {}

    for cfg in method_configs:
        exp_id = cfg["id"]
        # Sinh chuỗi hội tụ
        rounds = list(range(1, 31))
        initial_val = cfg["err_scale"] * 2.2
        target_val = cfg["err_scale"]
        val_losses = [
            float(target_val + (initial_val - target_val) * np.exp(-0.15 * r) + np.random.normal(0, 0.04))
            for r in rounds
        ]
        history_path = os.path.join(log_dir, f"{exp_id}_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({"rounds": rounds, "val_losses": val_losses}, f, indent=4)
        convergence_histories[cfg["algorithm"] + f" ({cfg['mode']})"] = {
            "rounds": rounds,
            "val_losses": val_losses,
        }

        # Sinh dự báo tương ứng
        pred_noise = np.random.normal(0, cfg["err_scale"], size=y_true_real.shape)
        # Bổ sung sai số cho 10% nút dị biệt
        tail_nodes = np.arange(int(num_nodes * 0.9), num_nodes)
        pred_noise[:, :, tail_nodes] *= cfg["worst_tail_boost"]
        y_pred = np.clip(y_true_real + pred_noise, 0.0, 80.0)

        np.savez_compressed(
            os.path.join(log_dir, f"{exp_id}_predictions.npz"),
            y_true=y_true_real,
            y_pred=y_pred,
            mean=mean_speed,
            std=std_speed,
        )

        display_name = f"{cfg['algorithm']} ({cfg['mode']})"
        predictions_dict[display_name] = y_pred

        # Tính toán kết quả theo horizon
        h_results = evaluate_experiment_horizons(y_true_real, y_pred)
        all_experiments.append({
            "metadata": cfg,
            "horizon_results": h_results,
        })

        node_stats = compute_per_node_errors(y_true_real, y_pred)
        node_mae_dict[display_name] = node_stats["mae"]

    # 1. Xuất bảng tổng hợp Markdown + CSV
    summary_df = build_benchmark_summary_dataframe(all_experiments)
    export_benchmark_reports(summary_df, report_dir=report_dir)

    # 2. Vẽ biểu đồ hội tụ so sánh
    plot_comparative_convergence(
        convergence_histories,
        save_path=os.path.join(report_dir, "convergence_comparison.png"),
        title="Đường Cong Hội Tụ So Sánh: Centralized vs Local vs FedAvg vs FedProx"
    )

    # 3. Vẽ biểu đồ chuỗi thời gian tại 3 trạm đại diện
    # Chọn nút 10 (trục chính biến động), nút 150 (ngoại vi ổn định), nút 200 (nút khó dự báo)
    plot_time_series_comparison(
        y_true=y_true_real,
        predictions_dict=predictions_dict,
        sensor_indices=[10, 150, 200],
        sensor_labels=["Trạm Trục Chính Cao Tốc (#10)", "Trạm Đường Nhánh Ngoại Ô (#150)", "Trạm Lưu Lượng Dị Biệt (#200)"],
        save_path=os.path.join(report_dir, "prediction_vs_ground_truth.png"),
    )

    # 4. Vẽ biểu đồ phân bố sai số Fairness
    plot_fairness_error_distribution(
        node_mae_dict=node_mae_dict,
        save_path=os.path.join(report_dir, "fairness_error_distribution.png"),
    )
    print("-> Đã hoàn tất khởi tạo Mockup và sinh toàn bộ báo cáo Task 3.3 thành công!")


if __name__ == "__main__":
    print("=== [Task 3.3] Chạy Thử Nghiệm Module Phân Tích & Báo Cáo ===")
    generate_mock_benchmark_artifacts(log_dir="logs", report_dir="reports")
