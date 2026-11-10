import os
import json
import numpy as np
import pytest
from src.utils.seed import set_seed
from src.data.partition import (
    partition_nodes_spatial_clustering,
    generate_spatial_skew_scenario,
    generate_temporal_shift_scenario,
    compute_distribution_divergence,
    plot_non_iid_distribution
)


def test_partition_nodes_spatial_clustering():
    set_seed(42)
    num_nodes = 50
    num_clients = 5

    # Tạo ma trận kề có cấu trúc 5 khối liên thông dày đặc (5 cụm rõ rệt)
    adj = np.zeros((num_nodes, num_nodes))
    nodes_per_cluster = num_nodes // num_clients
    for c in range(num_clients):
        start = c * nodes_per_cluster
        end = (c + 1) * nodes_per_cluster
        # Trong cụm có kết nối mạnh
        adj[start:end, start:end] = 1.0

    # Chạy gom cụm
    clusters = partition_nodes_spatial_clustering(adj, num_clients=num_clients, random_state=42)

    assert len(clusters) == num_clients, "Số lượng cụm trả về không khớp num_clients"

    all_nodes = []
    for cid, s_ids in clusters.items():
        assert len(s_ids) > 0, f"Client {cid} không được rỗng"
        all_nodes.extend(s_ids)

    # Đảm bảo phân hoạch bao phủ toàn bộ và không trùng lặp
    assert sorted(all_nodes) == list(range(num_nodes)), "Các sensor phân bổ không khớp tập {0..N-1}"


def test_spatial_skew_scenario():
    set_seed(42)
    num_nodes = 50
    num_clients = 5
    num_samples = 1000

    # Giả lập dữ liệu traffic (Time, Nodes) với sự phân hóa lưu lượng rõ rệt
    # Sensor 0..9: lưu lượng cao (80-100)
    # Sensor 10..19: lưu lượng khá (60-80)
    # Sensor 20..29: lưu lượng trung bình (40-60)
    # Sensor 30..39: lưu lượng thấp (20-40)
    # Sensor 40..49: lưu lượng rất thấp (5-20)
    traffic_data = np.zeros((num_samples, num_nodes))
    for c in range(num_clients):
        base_flow = 80 - c * 18
        start = c * 10
        end = (c + 1) * 10
        traffic_data[:, start:end] = np.random.normal(loc=base_flow, scale=4.0, size=(num_samples, 10))

    # Giả lập gom cụm ban đầu
    initial_clusters = {c: list(range(c * 10, (c + 1) * 10)) for c in range(num_clients)}

    out_json = "splits/spatial_skew.json"
    result = generate_spatial_skew_scenario(
        traffic_data=traffic_data,
        client_clusters=initial_clusters,
        output_path=out_json,
        num_samples=num_samples
    )

    assert os.path.exists(out_json), f"Không tìm thấy file {out_json}"
    assert result["scenario"] == "spatial_skew"
    assert result["num_clients"] == num_clients
    assert result["total_sensors"] == num_nodes
    assert result["total_samples"] == num_samples

    # Kiểm tra tính chất Skew: Client 0 phải có lưu lượng cao nhất, giảm dần về Client 4
    means = [result["clients"][str(k)]["mean_flow"] for k in range(num_clients)]
    for i in range(len(means) - 1):
        assert means[i] >= means[i + 1], f"Thứ tự lưu lượng không giảm dần: {means}"

    # Kiểm tra đo lường khoảng cách phân phối Wasserstein
    divergence_info = compute_distribution_divergence(result, traffic_data)
    assert divergence_info["mean_divergence"] > 0.0, "Độ lệch phân phối phải > 0"
    print(f"-> Spatial Skew Mean Wasserstein Divergence: {divergence_info['mean_divergence']}")

    # Kiểm tra xuất biểu đồ phân bố
    plot_path = "splits/spatial_skew_dist.png"
    plot_non_iid_distribution(result, traffic_data, save_path=plot_path)
    assert os.path.exists(plot_path), f"Không tìm thấy file biểu đồ {plot_path}"
    assert os.path.getsize(plot_path) > 1000, "File biểu đồ bị rỗng"


def test_temporal_shift_scenario():
    set_seed(42)
    num_samples = 2880  # 10 ngày (288 bước/ngày)
    num_clients = 5
    num_nodes = 307
    out_json = "splits/temporal_shift.json"

    result = generate_temporal_shift_scenario(
        num_samples=num_samples,
        num_clients=num_clients,
        output_path=out_json,
        num_nodes=num_nodes,
        samples_per_day=288
    )

    assert os.path.exists(out_json), f"Không tìm thấy file {out_json}"
    assert result["scenario"] == "temporal_shift"
    assert result["num_clients"] == num_clients

    # Kiểm tra tổng số mẫu của tất cả client phải bằng đúng num_samples (không rò rỉ, không mất mẫu)
    total_assigned_samples = sum(len(c["sample_indices"]) for c in result["clients"].values())
    assert total_assigned_samples == num_samples, f"Tổng số mẫu phân bổ ({total_assigned_samples}) != {num_samples}"

    # Kiểm tra không có mẫu nào bị trùng lặp giữa các client
    all_sample_indices = []
    for c in result["clients"].values():
        all_sample_indices.extend(c["sample_indices"])
    assert len(set(all_sample_indices)) == num_samples, "Có mẫu bị gán trùng lặp giữa các client"

    # Giả lập chuỗi thời gian traffic với biến thiên theo ngày (sáng/chiều cao điểm, đêm thấp điểm)
    t = np.arange(num_samples)
    day_cycle = np.sin(2 * np.pi * (t % 288) / 288 - np.pi / 2)  # Đỉnh vào giữa ngày, đáy vào ban đêm
    traffic_dummy = np.tile((day_cycle * 30 + 50)[:, None], (1, num_nodes))

    divergence_info = compute_distribution_divergence(result, traffic_dummy)
    assert divergence_info["mean_divergence"] > 0.0, "Độ lệch Temporal Shift phải > 0"
    print(f"-> Temporal Shift Mean Wasserstein Divergence: {divergence_info['mean_divergence']}")

    # Kiểm tra xuất biểu đồ phân bố
    plot_path = "splits/temporal_shift_dist.png"
    plot_non_iid_distribution(result, traffic_dummy, save_path=plot_path)
    assert os.path.exists(plot_path), f"Không tìm thấy file biểu đồ {plot_path}"
    assert os.path.getsize(plot_path) > 1000, "File biểu đồ bị rỗng"


if __name__ == "__main__":
    print("=== Chạy kiểm thử cho Task 1.3: Non-IID Partition ===")
    test_partition_nodes_spatial_clustering()
    print("-> test_partition_nodes_spatial_clustering PASSED")

    test_spatial_skew_scenario()
    print("-> test_spatial_skew_scenario PASSED")

    test_temporal_shift_scenario()
    print("-> test_temporal_shift_scenario PASSED")

    print("\n>>> TẤT CẢ UNIT TESTS CHO TASK 1.3 ĐÃ HOÀN THÀNH XUẤT SẮC! <<<")

