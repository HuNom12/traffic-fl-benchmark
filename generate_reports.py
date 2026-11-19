import os
import sys
import argparse
import glob
import json
import numpy as np
import pandas as pd

# Đảm bảo đường dẫn import src hoạt động đúng
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.post_process import (
    evaluate_experiment_horizons,
    build_benchmark_summary_dataframe,
    export_benchmark_reports,
    plot_comparative_convergence,
    plot_time_series_comparison,
    plot_fairness_error_distribution,
    compute_per_node_errors,
    generate_mock_benchmark_artifacts,
)


def run_reports_pipeline(log_dir: str = "logs", report_dir: str = "reports", force_mock: bool = False):
    """Quét và phân tích toàn bộ kết quả huấn luyện từ log_dir, xuất báo cáo ra report_dir."""
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pred_files = glob.glob(os.path.join(log_dir, "*_predictions.npz"))

    if len(pred_files) == 0 or force_mock:
        print("\n[THÔNG BÁO] Chưa tìm thấy file log thực nghiệm thật trong logs/ hoặc cờ --mock được kích hoạt.")
        print("-> Tự động kích hoạt cơ chế Mockup Artifacts để kiểm thử toàn diện pipeline Task 3.3...")
        generate_mock_benchmark_artifacts(
            log_dir=log_dir,
            report_dir=report_dir,
            num_nodes=207,
            num_samples=288,
        )
        print(f"\n>>> Báo cáo và biểu đồ mẫu đã được lưu thành công tại: {report_dir}/ <<<")
        return

    print(f"\n[XỬ LÝ] Tìm thấy {len(pred_files)} file kết quả thực nghiệm trong {log_dir}/:")
    for f in pred_files:
        print(f"  + {os.path.basename(f)}")

    experiments_data = []
    predictions_dict = {}
    node_mae_dict = {}
    convergence_histories = {}
    reference_y_true = None

    for pfile in pred_files:
        base_name = os.path.basename(pfile).replace("_predictions.npz", "")
        data = np.load(pfile)
        y_true = data["y_true"]
        y_pred = data["y_pred"]

        if reference_y_true is None:
            reference_y_true = y_true

        # Đọc metadata hoặc suy ra từ tên file
        history_file = os.path.join(log_dir, f"{base_name}_history.json")
        meta = {
            "model": "STGCN" if "stgcn" in base_name.lower() else "GRU",
            "mode": "centralized" if "centralized" in base_name.lower() else ("local" if "local" in base_name.lower() else "federated"),
            "algorithm": "FedProx" if "fedprox" in base_name.lower() else ("FedAvg" if "fedavg" in base_name.lower() else "Adam"),
            "scenario": "temporal_shift" if "temporal" in base_name.lower() else ("spatial_skew" if "spatial" in base_name.lower() else "global"),
        }

        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as hf:
                    hdata = json.load(hf)
                    meta.update(hdata.get("metadata", {}))
                    convergence_histories[base_name] = hdata
            except Exception as e:
                print(f"  ! Không thể đọc lịch sử từ {history_file}: {e}")

        h_results = evaluate_experiment_horizons(y_true, y_pred)
        experiments_data.append({
            "metadata": meta,
            "horizon_results": h_results,
        })

        display_name = f"{meta['algorithm']} ({meta['mode']})"
        predictions_dict[display_name] = y_pred
        node_stats = compute_per_node_errors(y_true, y_pred)
        node_mae_dict[display_name] = node_stats["mae"]

    # 1. Xuất bảng Benchmark CSV & Markdown
    summary_df = build_benchmark_summary_dataframe(experiments_data)
    export_benchmark_reports(summary_df, report_dir=report_dir)

    # 2. Vẽ biểu đồ hội tụ nếu có lịch sử
    if len(convergence_histories) > 0:
        plot_comparative_convergence(
            convergence_histories,
            save_path=os.path.join(report_dir, "convergence_comparison.png"),
            title="Đường Cong Hội Tụ So Sánh Giữa Các Chế Độ Học",
        )

    # 3. Vẽ biểu đồ chuỗi thời gian nếu có ít nhất 1 dự báo
    if reference_y_true is not None and len(predictions_dict) > 0:
        # Chọn 3 node tiêu biểu
        num_nodes = reference_y_true.shape[2]
        sample_nodes = [0, num_nodes // 2, num_nodes - 1]
        plot_time_series_comparison(
            y_true=reference_y_true,
            predictions_dict=predictions_dict,
            sensor_indices=sample_nodes,
            sensor_labels=[f"Sensor #{sample_nodes[0]}", f"Sensor #{sample_nodes[1]}", f"Sensor #{sample_nodes[2]}"],
            save_path=os.path.join(report_dir, "prediction_vs_ground_truth.png"),
        )

    # 4. Vẽ biểu đồ phân bố sai số Fairness
    if len(node_mae_dict) > 0:
        plot_fairness_error_distribution(
            node_mae_dict=node_mae_dict,
            save_path=os.path.join(report_dir, "fairness_error_distribution.png"),
        )

    print(f"\n>>> Hoàn thành phân tích và trực quan hóa Task 3.3! Kết quả đã lưu vào: {report_dir}/ <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3.3: Benchmark Reports & Visualization Pipeline")
    parser.add_argument("--log-dir", type=str, default="logs", help="Thư mục chứa các tệp log từ Task 3.1")
    parser.add_argument("--report-dir", type=str, default="reports", help="Thư mục xuất báo cáo và biểu đồ")
    parser.add_argument("--mock", action="store_true", help="Bắt buộc chạy với dữ liệu giả lập mẫu")
    args = parser.parse_args()

    run_reports_pipeline(log_dir=args.log_dir, report_dir=args.report_dir, force_mock=args.mock)
