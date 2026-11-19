import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.utils.post_process import (
    compute_per_node_errors,
    compute_fairness_metrics,
    evaluate_experiment_horizons,
    build_benchmark_summary_dataframe,
    export_benchmark_reports,
    plot_comparative_convergence,
    plot_time_series_comparison,
    plot_fairness_error_distribution,
    generate_mock_benchmark_artifacts,
)


def test_per_node_and_fairness_metrics():
    num_samples = 100
    time_out = 12
    num_nodes = 50

    # Giả lập y_true từ 40 đến 70 mph
    y_true = np.random.uniform(40.0, 70.0, size=(num_samples, time_out, num_nodes))
    # Giả lập y_pred với độ lệch tăng dần theo node_id (node 0 sai số thấp, node 49 sai số cao)
    noise = np.zeros_like(y_true)
    for v in range(num_nodes):
        noise[:, :, v] = np.random.normal(0.0, 1.0 + v * 0.1, size=(num_samples, time_out))
    y_pred = y_true + noise

    # 1. Tính toán Per-Node MAE
    node_stats = compute_per_node_errors(y_true, y_pred, horizon_idx=2)
    mae_arr = node_stats["mae"]
    assert len(mae_arr) == num_nodes
    assert np.all(np.isfinite(mae_arr))

    # Node cuối phải có sai số lớn hơn node đầu
    assert np.mean(mae_arr[:10]) < np.mean(mae_arr[-10:])

    # 2. Tính toán Fairness / Worst-Decile
    fairness = compute_fairness_metrics(mae_arr, decile_ratio=0.10)
    assert fairness["worst_node_mae"] >= fairness["worst_decile_mae"]
    assert fairness["worst_decile_mae"] > fairness["best_decile_mae"]
    assert fairness["fairness_gap"] > 1.0
    print(
        f"-> Per-Node & Fairness test: Mean MAE={fairness['mean_mae']:.2f}, "
        f"Worst-Decile={fairness['worst_decile_mae']:.2f}, Gap={fairness['fairness_gap']:.2f} [PASSED]"
    )


def test_horizon_evaluation_and_table_generation():
    num_samples = 50
    time_out = 12
    num_nodes = 20

    y_true = np.random.uniform(30.0, 60.0, size=(num_samples, time_out, num_nodes))
    y_pred = y_true + np.random.normal(0.0, 3.0, size=y_true.shape)

    h_results = evaluate_experiment_horizons(y_true, y_pred)
    assert len(h_results) == 4  # 15m, 30m, 60m, Overall
    horizons_found = [h["horizon"] for h in h_results]
    assert "15 min" in horizons_found
    assert "30 min" in horizons_found
    assert "60 min" in horizons_found
    assert "Overall" in horizons_found

    exp_data = [{
        "metadata": {
            "model": "STGCN",
            "mode": "federated",
            "algorithm": "FedProx",
            "scenario": "spatial_skew",
        },
        "horizon_results": h_results,
    }]

    df = build_benchmark_summary_dataframe(exp_data)
    assert len(df) == 4
    assert "MAE (mph)" in df.columns
    assert "Worst-Decile MAE" in df.columns

    csv_path, md_path = export_benchmark_reports(df, report_dir="reports/test_reports")
    assert os.path.exists(csv_path)
    assert os.path.exists(md_path)
    print("-> Horizon evaluation & Report export [PASSED]")


def test_plotting_functions():
    test_dir = "reports/test_plots"
    os.makedirs(test_dir, exist_ok=True)

    # 1. Test Convergence Plot
    histories = {
        "FedAvg": {"rounds": [1, 2, 3], "val_losses": [5.0, 4.0, 3.5]},
        "FedProx": {"rounds": [1, 2, 3], "val_losses": [4.8, 3.8, 3.2]},
    }
    conv_path = os.path.join(test_dir, "test_conv.png")
    plot_comparative_convergence(histories, save_path=conv_path)
    assert os.path.exists(conv_path) and os.path.getsize(conv_path) > 1000

    # 2. Test Time-Series Plot
    y_true = np.random.uniform(40.0, 60.0, size=(288, 12, 10))
    preds = {
        "FedAvg": y_true + np.random.normal(0, 3, size=y_true.shape),
        "FedProx": y_true + np.random.normal(0, 2, size=y_true.shape),
    }
    ts_path = os.path.join(test_dir, "test_ts.png")
    plot_time_series_comparison(y_true, preds, sensor_indices=[0, 5], save_path=ts_path)
    assert os.path.exists(ts_path) and os.path.getsize(ts_path) > 1000

    # 3. Test Fairness Boxplot
    node_mae = {
        "FedAvg": np.random.uniform(2.5, 6.0, size=50),
        "FedProx": np.random.uniform(2.3, 4.8, size=50),
    }
    box_path = os.path.join(test_dir, "test_box.png")
    plot_fairness_error_distribution(node_mae, save_path=box_path)
    assert os.path.exists(box_path) and os.path.getsize(box_path) > 1000

    print("-> Plotting functions [PASSED]")


if __name__ == "__main__":
    print("=== [Test Task 3.3] Kiểm Thử Kỹ Thuật Module Báo Cáo & Trực Quan Hóa ===")
    test_per_node_and_fairness_metrics()
    test_horizon_evaluation_and_table_generation()
    test_plotting_functions()
    print("\n>>> TẤT CẢ UNIT TESTS CHO TASK 3.3 ĐÃ HOÀN THÀNH XUẤT SẮC! <<<")
