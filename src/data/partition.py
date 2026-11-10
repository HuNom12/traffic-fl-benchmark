import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def partition_nodes_spatial_clustering(
    adj_matrix: np.ndarray,
    num_clients: int = 5,
    random_state: int = 42
) -> Dict[int, List[int]]:
    """Phân chia N cảm biến về K Client/Node dựa trên đồ thị không gian.

    Sử dụng Spectral Clustering trên ma trận kề đối xứng. Trong trường hợp
    ma trận kề thưa hoặc đồ thị suy biến (ví dụ ma trận đơn vị), tự động chuyển sang
    phân chia phổ thông qua ma trận Laplacian hoặc gom theo phân đoạn để đảm bảo tính ổn định.

    Args:
        adj_matrix: Ma trận kề đồ thị kích thước (N, N).
        num_clients: Số lượng Client cần phân chia (K).
        random_state: Seed ngẫu nhiên để tái lập kết quả.

    Returns:
        Dict: {client_id: [danh_sách_sensor_id_đã_sắp_xếp]}
    """
    num_nodes = adj_matrix.shape[0]
    if num_clients > num_nodes:
        raise ValueError(
            f"Số client ({num_clients}) không được lớn hơn số lượng cảm biến ({num_nodes})"
        )

    # Đảm bảo ma trận kề đối xứng và không có giá trị âm
    w = (adj_matrix + adj_matrix.T) / 2.0
    w = np.clip(w, 0.0, None)
    np.fill_diagonal(w, 0.0)

    # Kiểm tra xem đồ thị có cạnh nào hay không
    total_edges = np.sum(w > 0)
    if total_edges == 0:
        # Trường hợp ma trận kề rỗng hoặc đơn vị: chia đều theo chỉ số
        indices = np.arange(num_nodes)
        split_indices = np.array_split(indices, num_clients)
        return {k: split_indices[k].tolist() for k in range(num_clients)}

    try:
        # Thử nghiệm với Spectral Clustering trực tiếp
        clustering = SpectralClustering(
            n_clusters=num_clients,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
            n_init=10
        )
        labels = clustering.fit_predict(w)
    except Exception:
        # Fallback an toàn: Tính toán Normalized Laplacian và áp dụng KMeans trên Eigenvectors
        degrees = np.sum(w, axis=1)
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
        d_inv_sqrt[degrees == 0] = 0.0
        d_mat = np.diag(d_inv_sqrt)
        laplacian_norm = np.eye(num_nodes) - d_mat @ w @ d_mat

        # Tính giá trị riêng và vector riêng
        evals, evecs = np.linalg.eigh(laplacian_norm)
        # Lấy num_clients vector riêng ứng với các giá trị riêng nhỏ nhất
        k_evecs = evecs[:, :num_clients]

        # Chuẩn hóa hàng
        norms = np.linalg.norm(k_evecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_features = k_evecs / norms

        kmeans = KMeans(n_clusters=num_clients, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(normalized_features)

    # Gom nhóm theo client_id
    clusters: Dict[int, List[int]] = {k: [] for k in range(num_clients)}
    for node_idx, label in enumerate(labels):
        clusters[int(label)].append(int(node_idx))

    # Đảm bảo không có client nào bị rỗng (xử lý corner-case)
    empty_clients = [k for k, v in clusters.items() if len(v) == 0]
    if empty_clients:
        for empty_k in empty_clients:
            largest_k = max(clusters.keys(), key=lambda k: len(clusters[k]))
            borrowed_node = clusters[largest_k].pop()
            clusters[empty_k].append(borrowed_node)

    return {k: sorted(v) for k, v in clusters.items()}


def _extract_node_mean_flow(traffic_data: np.ndarray) -> np.ndarray:
    """Trích xuất lưu lượng trung bình theo từng nút cảm biến.

    Hỗ trợ shape 2D (Time, Nodes), 3D (Time, Nodes, Features) và 4D (Samples, Time, Nodes, Features).
    """
    if traffic_data.ndim == 4:
        # traffic_data: (Samples, Time, Nodes, Features) -> lấy kênh flow (f=0)
        return np.mean(traffic_data[:, :, :, 0], axis=(0, 1))
    elif traffic_data.ndim == 3:
        # traffic_data: (Time, Nodes, Features) -> lấy kênh flow (f=0)
        return np.mean(traffic_data[:, :, 0], axis=0)
    elif traffic_data.ndim == 2:
        # traffic_data: (Time, Nodes)
        return np.mean(traffic_data, axis=0)
    else:
        raise ValueError(f"Shape dữ liệu {traffic_data.shape} không được hỗ trợ (chỉ nhận 2D, 3D, 4D)")


def generate_spatial_skew_scenario(
    traffic_data: np.ndarray,
    client_clusters: Dict[int, List[int]],
    output_path: str = "splits/spatial_skew.json",
    num_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Kịch bản Non-IID 1: Spatial Skew / Feature Skew.

    Sắp xếp và phân bổ các cụm cảm biến sao cho tạo ra sự chênh lệch phân phối lưu lượng rõ rệt:
    - Client 0 sở hữu cụm cảm biến có lưu lượng trung bình cao nhất (nội đô / trục đường chính).
    - Client (K - 1) sở hữu cụm cảm biến có lưu lượng thấp nhất (ngoại ô / đường nhánh).

    Args:
        traffic_data: Mảng lưu lượng giao thông (2D, 3D hoặc 4D).
        client_clusters: Dict gom cụm {cluster_id: [sensor_ids]}.
        output_path: Đường dẫn lưu file cấu hình phân chia JSON.
        num_samples: Tổng số mẫu thời gian (nếu None sẽ suy ra từ chiều đầu tiên của traffic_data).

    Returns:
        Dict chứa thông tin cấu hình phân chia chi tiết của từng Client.
    """
    node_mean_flow = _extract_node_mean_flow(traffic_data)
    num_nodes = len(node_mean_flow)
    num_clients = len(client_clusters)

    if num_samples is None:
        total_samples = traffic_data.shape[0]
    else:
        total_samples = int(num_samples)

    # 1. Tính toán lưu lượng trung bình của từng cụm
    cluster_stats = []
    for cluster_id, sensor_list in client_clusters.items():
        if len(sensor_list) == 0:
            cluster_mean = 0.0
        else:
            cluster_mean = float(np.mean(node_mean_flow[sensor_list]))
        cluster_stats.append((cluster_id, cluster_mean, sensor_list))

    # 2. Sắp xếp các cụm theo lưu lượng giảm dần (High-flow -> Low-flow)
    cluster_stats.sort(key=lambda item: item[1], reverse=True)

    # 3. Gán lại client_id theo thứ tự chênh lệch
    client_splits: Dict[str, Any] = {}
    sample_indices = list(range(total_samples))

    for new_client_id, (_, _, sensor_list) in enumerate(cluster_stats):
        if len(sensor_list) > 0:
            sensor_flows = node_mean_flow[sensor_list]
            c_mean = float(np.mean(sensor_flows))
            c_std = float(np.std(sensor_flows))
        else:
            c_mean = 0.0
            c_std = 0.0

        client_splits[str(new_client_id)] = {
            "client_id": new_client_id,
            "sensor_ids": [int(s) for s in sorted(sensor_list)],
            "sample_indices": sample_indices,
            "num_sensors": len(sensor_list),
            "num_samples": total_samples,
            "mean_flow": round(c_mean, 4),
            "std_flow": round(c_std, 4)
        }

    scenario_result: Dict[str, Any] = {
        "scenario": "spatial_skew",
        "num_clients": num_clients,
        "total_sensors": num_nodes,
        "total_samples": total_samples,
        "clients": client_splits
    }

    save_split_indices(scenario_result, output_path)
    return scenario_result


def generate_temporal_shift_scenario(
    num_samples: int,
    num_clients: int = 5,
    output_path: str = "splits/temporal_shift.json",
    num_nodes: Optional[int] = None,
    samples_per_day: int = 288
) -> Dict[str, Any]:
    """Kịch bản Non-IID 2: Temporal Shift / Concept Drift.

    Phân chia các mẫu thời gian giữa các Client lệch theo các pha sinh hoạt trong ngày (24 giờ):
    - Client 0: Khung giờ đêm khuya / rạng sáng (Off-peak).
    - Client 1: Khung giờ sáng sớm & cao điểm sáng (Morning Rush).
    - Client 2: Khung giờ hành chính ban ngày (Daytime Normal).
    - Client 3: Khung giờ chiều & cao điểm chiều (Evening Rush).
    - Client 4 (nếu K=5): Khung giờ buổi tối (Evening / Late night).

    Toàn bộ các mẫu đều được sử dụng (không rò rỉ, không mất mẫu).

    Args:
        num_samples: Tổng số mẫu chuỗi thời gian (ví dụ trên tập Train).
        num_clients: Số Client tham gia (K=4 đến 8).
        output_path: Đường dẫn lưu file cấu hình JSON.
        num_nodes: Tổng số cảm biến (mỗi client sở hữu toàn bộ sensor nếu num_nodes != None).
        samples_per_day: Số bước lấy mẫu trong một chu kỳ ngày (mặc định 288 bước cho PeMS 5 phút).

    Returns:
        Dict cấu hình phân chia kịch bản Temporal Shift.
    """
    if num_clients < 2:
        raise ValueError("Số lượng client cho kịch bản Temporal Shift phải >= 2")

    # Xác định các ranh giới bước thời gian trong ngày cho K client
    step_boundaries = [int(round(k * samples_per_day / num_clients)) for k in range(num_clients + 1)]

    # Phân loại từng sample_idx vào đúng client dựa trên chu kỳ ngày
    client_samples: Dict[int, List[int]] = {k: [] for k in range(num_clients)}
    for sample_idx in range(num_samples):
        step_in_day = sample_idx % samples_per_day
        for k in range(num_clients):
            if step_boundaries[k] <= step_in_day < step_boundaries[k + 1]:
                client_samples[k].append(sample_idx)
                break

    all_sensor_ids = list(range(num_nodes)) if num_nodes is not None else []

    client_splits: Dict[str, Any] = {}
    for k in range(num_clients):
        start_step = step_boundaries[k]
        end_step = step_boundaries[k + 1]
        start_hour = (start_step * 24) / samples_per_day
        end_hour = (end_step * 24) / samples_per_day

        client_splits[str(k)] = {
            "client_id": k,
            "sensor_ids": all_sensor_ids,
            "sample_indices": client_samples[k],
            "num_sensors": len(all_sensor_ids),
            "num_samples": len(client_samples[k]),
            "diurnal_step_range": [start_step, end_step],
            "diurnal_hour_range": [round(start_hour, 2), round(end_hour, 2)]
        }

    scenario_result: Dict[str, Any] = {
        "scenario": "temporal_shift",
        "num_clients": num_clients,
        "total_sensors": num_nodes if num_nodes is not None else 0,
        "total_samples": num_samples,
        "samples_per_day": samples_per_day,
        "clients": client_splits
    }

    save_split_indices(scenario_result, output_path)
    return scenario_result


def save_split_indices(data: Dict[str, Any], filepath: str) -> None:
    """Lưu tệp index cấu hình cố định dạng JSON để đảm bảo tính tái lập."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"-> Đã lưu cấu hình phân chia tại: {filepath}")


def compute_distribution_divergence(
    split_info: Dict[str, Any],
    traffic_data: np.ndarray
) -> Dict[str, Any]:
    """Tính toán ma trận khoảng cách phân phối Wasserstein giữa các cặp Client.

    Chứng minh định lượng tính chất Non-IID giữa các Client.
    """
    node_mean_flow = _extract_node_mean_flow(traffic_data)
    clients_dict = split_info.get("clients", {})
    num_clients = len(clients_dict)
    client_ids = [int(k) for k in clients_dict.keys()]

    # Thu thập dữ liệu phân bố của từng client
    client_distributions = {}
    scenario = split_info.get("scenario", "")

    if scenario == "spatial_skew":
        for cid_str, info in clients_dict.items():
            s_ids = info["sensor_ids"]
            if len(s_ids) > 0:
                client_distributions[int(cid_str)] = node_mean_flow[s_ids]
            else:
                client_distributions[int(cid_str)] = np.array([0.0])
    else:  # temporal_shift
        # Lấy giá trị lưu lượng trung bình toàn mạng theo từng sample_idx
        if traffic_data.ndim in (3, 4):
            flow_series = np.mean(traffic_data[:, :, 0], axis=1) if traffic_data.ndim == 3 else np.mean(traffic_data[:, :, :, 0], axis=(1, 2))
        else:
            flow_series = np.mean(traffic_data, axis=1)

        for cid_str, info in clients_dict.items():
            sample_ids = info["sample_indices"]
            if len(sample_ids) > 0:
                client_distributions[int(cid_str)] = flow_series[sample_ids]
            else:
                client_distributions[int(cid_str)] = np.array([0.0])

    # Tính Wasserstein distance cho từng cặp (i, j)
    wasserstein_matrix = np.zeros((num_clients, num_clients))
    pairwise_distances = []

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = stats.wasserstein_distance(client_distributions[i], client_distributions[j])
            wasserstein_matrix[i, j] = dist
            wasserstein_matrix[j, i] = dist
            pairwise_distances.append({
                "pair": f"Client_{i} vs Client_{j}",
                "wasserstein_distance": round(float(dist), 4)
            })

    return {
        "scenario": scenario,
        "pairwise_wasserstein": pairwise_distances,
        "mean_divergence": round(float(np.mean([p["wasserstein_distance"] for p in pairwise_distances])), 4)
    }


def plot_non_iid_distribution(
    split_info: Dict[str, Any],
    traffic_data: np.ndarray,
    save_path: str = "splits/distribution_check.png"
) -> None:
    """Vẽ biểu đồ phân bố lưu lượng hoặc thời gian chứng minh dữ liệu bị lệch phân phối rõ rệt.

    - Với Spatial Skew: Vẽ đồ thị phân phối (KDE/Histogram) của lưu lượng xe qua các sensor của từng client.
    - Với Temporal Shift: Vẽ biểu đồ tần suất phân bổ mẫu theo các giờ trong ngày (0h - 24h).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scenario = split_info.get("scenario", "spatial_skew")
    clients_dict = split_info.get("clients", {})
    num_clients = len(clients_dict)

    plt.figure(figsize=(10, 6))

    if scenario == "spatial_skew":
        node_mean_flow = _extract_node_mean_flow(traffic_data)
        colors = plt.cm.viridis(np.linspace(0, 1, num_clients))

        for idx, (cid_str, info) in enumerate(clients_dict.items()):
            s_ids = info["sensor_ids"]
            if len(s_ids) > 0:
                flows = node_mean_flow[s_ids]
                plt.hist(
                    flows,
                    bins=15,
                    alpha=0.5,
                    density=True,
                    label=f"Client {cid_str} (mean={info['mean_flow']:.1f}, N={len(s_ids)})",
                    color=colors[idx]
                )

        plt.title("Phân bố lưu lượng giữa các Client (Spatial Skew / Feature Skew)", fontsize=13)
        plt.xlabel("Lưu lượng trung bình (Mean Traffic Flow)", fontsize=11)
        plt.ylabel("Mật độ xác suất (Density)", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    elif scenario == "temporal_shift":
        samples_per_day = split_info.get("samples_per_day", 288)
        colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

        for idx, (cid_str, info) in enumerate(clients_dict.items()):
            sample_ids = np.array(info["sample_indices"])
            if len(sample_ids) > 0:
                hours = (sample_ids % samples_per_day) * 24.0 / samples_per_day
                plt.hist(
                    hours,
                    bins=24,
                    range=(0, 24),
                    alpha=0.6,
                    label=f"Client {cid_str} ({info['diurnal_hour_range'][0]}h - {info['diurnal_hour_range'][1]}h, M={len(sample_ids)})",
                    color=colors[idx]
                )

        plt.title("Phân bổ số lượng mẫu theo 24 giờ trong ngày (Temporal Shift / Concept Drift)", fontsize=13)
        plt.xlabel("Giờ trong ngày (0h - 24h)", fontsize=11)
        plt.ylabel("Số lượng mẫu (Sample Count)", fontsize=11)
        plt.xticks(range(0, 25, 2))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố tại: {save_path}")



def compute_distribution_divergence(
    split_info: Dict[str, Any],
    traffic_data: np.ndarray
) -> Dict[str, Any]:
    """Tính toán ma trận khoảng cách phân phối Wasserstein giữa các cặp Client.

    Chứng minh định lượng tính chất Non-IID giữa các Client.
    """
    node_mean_flow = _extract_node_mean_flow(traffic_data)
    clients_dict = split_info.get("clients", {})
    num_clients = len(clients_dict)
    client_ids = [int(k) for k in clients_dict.keys()]

    # Thu thập dữ liệu phân bố của từng client
    client_distributions = {}
    scenario = split_info.get("scenario", "")

    if scenario == "spatial_skew":
        for cid_str, info in clients_dict.items():
            s_ids = info["sensor_ids"]
            if len(s_ids) > 0:
                client_distributions[int(cid_str)] = node_mean_flow[s_ids]
            else:
                client_distributions[int(cid_str)] = np.array([0.0])
    else:  # temporal_shift
        # Lấy giá trị lưu lượng trung bình toàn mạng theo từng sample_idx
        if traffic_data.ndim in (3, 4):
            flow_series = np.mean(traffic_data[:, :, 0], axis=1) if traffic_data.ndim == 3 else np.mean(traffic_data[:, :, :, 0], axis=(1, 2))
        else:
            flow_series = np.mean(traffic_data, axis=1)

        for cid_str, info in clients_dict.items():
            sample_ids = info["sample_indices"]
            if len(sample_ids) > 0:
                client_distributions[int(cid_str)] = flow_series[sample_ids]
            else:
                client_distributions[int(cid_str)] = np.array([0.0])

    # Tính Wasserstein distance cho từng cặp (i, j)
    wasserstein_matrix = np.zeros((num_clients, num_clients))
    pairwise_distances = []

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = stats.wasserstein_distance(client_distributions[i], client_distributions[j])
            wasserstein_matrix[i, j] = dist
            wasserstein_matrix[j, i] = dist
            pairwise_distances.append({
                "pair": f"Client_{i} vs Client_{j}",
                "wasserstein_distance": round(float(dist), 4)
            })

    return {
        "scenario": scenario,
        "pairwise_wasserstein": pairwise_distances,
        "mean_divergence": round(float(np.mean([p["wasserstein_distance"] for p in pairwise_distances])), 4)
    }


def plot_non_iid_distribution(
    split_info: Dict[str, Any],
    traffic_data: np.ndarray,
    save_path: str = "splits/distribution_check.png"
) -> None:
    """Vẽ biểu đồ phân bố lưu lượng hoặc thời gian chứng minh dữ liệu bị lệch phân phối rõ rệt.

    - Với Spatial Skew: Vẽ đồ thị phân phối (KDE/Histogram) của lưu lượng xe qua các sensor của từng client.
    - Với Temporal Shift: Vẽ biểu đồ tần suất phân bổ mẫu theo các giờ trong ngày (0h - 24h).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scenario = split_info.get("scenario", "spatial_skew")
    clients_dict = split_info.get("clients", {})
    num_clients = len(clients_dict)

    plt.figure(figsize=(10, 6))

    if scenario == "spatial_skew":
        node_mean_flow = _extract_node_mean_flow(traffic_data)
        colors = plt.cm.viridis(np.linspace(0, 1, num_clients))

        for idx, (cid_str, info) in enumerate(clients_dict.items()):
            s_ids = info["sensor_ids"]
            if len(s_ids) > 0:
                flows = node_mean_flow[s_ids]
                plt.hist(
                    flows,
                    bins=15,
                    alpha=0.5,
                    density=True,
                    label=f"Client {cid_str} (mean={info['mean_flow']:.1f}, N={len(s_ids)})",
                    color=colors[idx]
                )

        plt.title("Phân bố lưu lượng giữa các Client (Spatial Skew / Feature Skew)", fontsize=13)
        plt.xlabel("Lưu lượng trung bình (Mean Traffic Flow)", fontsize=11)
        plt.ylabel("Mật độ xác suất (Density)", fontsize=11)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    elif scenario == "temporal_shift":
        samples_per_day = split_info.get("samples_per_day", 288)
        colors = plt.cm.tab10(np.linspace(0, 1, num_clients))

        for idx, (cid_str, info) in enumerate(clients_dict.items()):
            sample_ids = np.array(info["sample_indices"])
            if len(sample_ids) > 0:
                hours = (sample_ids % samples_per_day) * 24.0 / samples_per_day
                plt.hist(
                    hours,
                    bins=24,
                    range=(0, 24),
                    alpha=0.6,
                    label=f"Client {cid_str} ({info['diurnal_hour_range'][0]}h - {info['diurnal_hour_range'][1]}h, M={len(sample_ids)})",
                    color=colors[idx]
                )

        plt.title("Phân bổ số lượng mẫu theo 24 giờ trong ngày (Temporal Shift / Concept Drift)", fontsize=13)
        plt.xlabel("Giờ trong ngày (0h - 24h)", fontsize=11)
        plt.ylabel("Số lượng mẫu (Sample Count)", fontsize=11)
        plt.xticks(range(0, 25, 2))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố tại: {save_path}")