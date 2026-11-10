# Traffic FL Benchmark 🚦📡

Hệ thống Benchmark Federated Learning (Học liên kết) cho bài toán dự báo chuỗi thời gian không-thời gian trên mạng lưới giao thông (Spatio-Temporal Traffic Forecasting trên tập dữ liệu PeMS).

---

## 1. Cấu trúc Dự án

```text
traffic-fl-benchmark/
├── configs/                  # File cấu hình trung tâm (YAML)
│   ├── config.yaml           # Cấu hình siêu tham số toàn hệ thống
│   ├── dataset/              # Cấu hình tập dữ liệu (pems04.yaml)
│   └── non_iid/              # Cấu hình các kịch bản Non-IID (spatial_skew.yaml, temporal_shift.yaml)
├── data/
│   ├── raw/                  # Dữ liệu gốc (.npz, distance.csv)
│   └── processed/            # Dữ liệu sau khi xử lý & chuẩn hóa
├── splits/                   # Chỉ số phân chia dữ liệu Non-IID (.json) và biểu đồ phân phối (.png)
├── src/
│   ├── data/
│   │   ├── dataset.py        # Tiền xử lý PeMS, Z-score, sliding window, ma trận kề A
│   │   └── partition.py      # Phân rã đồ thị K-client, sinh kịch bản Spatial Skew & Temporal Shift
│   ├── models/
│   │   ├── base.py           # BaseForecaster trừu tượng (chuẩn hóa tensor 4D)
│   │   ├── temporal.py       # GRU / LSTM
│   │   └── spatio_temporal.py# STGCN
│   ├── federated/
│   │   ├── client.py         # ClientNode (hỗ trợ FedAvg & FedProx)
│   │   ├── server.py         # FederatedServer (Tổng hợp mô hình FedAvg)
│   │   └── trainer.py        # Quản lý vòng lặp huấn luyện FL
│   └── utils/
│       ├── seed.py           # Khóa seed ngẫu nhiên (Reproducibility)
│       └── metrics.py        # Các hàm đánh giá (MAE, RMSE, MAPE)
└── tests/                    # Bộ kiểm thử đơn vị mô-đun
    ├── test_fl_core.py       # Kiểm thử hạ tầng FL, ClientNode, FedAvg, FedProx
    ├── test_data_pipeline.py # Kiểm thử tiền xử lý PeMS và ma trận kề A (Task 1.2)
    ├── test_patrition.py     # Kiểm thử kịch bản Non-IID tích hợp với PeMS pipeline
    └── test_partition.py    # Kiểm thử độc lập thuật toán phân rã Non-IID và vẽ biểu đồ
```

---

## 2. Hướng dẫn Cài đặt Môi trường

Yêu cầu Python >= 3.10. Cài đặt các thư viện phụ thuộc:

```bash
pip install -r requirements.txt
```

---

## 3. Hướng dẫn Chạy Kiểm thử (Running Tests)

> [!IMPORTANT]
> Khi chạy các script kiểm thử từ thư mục gốc, cần thiết lập `PYTHONPATH=.` để Python nhận diện được package `src`.
> Trên Windows PowerShell, khuyến nghị thiết lập thêm `PYTHONIOENCODING="utf-8"` để hiển thị đúng ký tự tiếng Việt.

### 3.1. Chạy trên Windows (PowerShell)

#### Test 1: Kiểm thử Hạ tầng FL Cốt lõi (Task 1.1)
Kiểm tra khả năng tái lập (Seed), interface `BaseForecaster`, kích thước tensor 4D, huấn luyện `ClientNode` (cả FedAvg và FedProx) và tổng hợp `FederatedServer`:
```powershell
$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"; python tests/test_fl_core.py
```

#### Test 2: Kiểm thử Pipeline Tiền xử lý Dữ liệu PeMS (Task 1.2)
Kiểm tra hàm tính ma trận kề ngưỡng Gaussian $A$, phân chia Train/Val/Test tuần tự, Z-score và tạo Sliding Window:
```powershell
$env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"; python tests/test_data_pipeline.py
```
*(Lưu ý: Yêu cầu hoàn thiện Task 1.2 trong `src/data/dataset.py`)*

#### Test 3: Kiểm thử Kịch bản Phân chia Non-IID (Task 1.3)
* **Chạy kiểm thử tích hợp:**
  ```powershell
  $env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"; python tests/test_patrition.py
  ```
* **Chạy kiểm thử độc lập & sinh biểu đồ phân bố minh chứng:**
  ```powershell
  $env:PYTHONPATH="."; $env:PYTHONIOENCODING="utf-8"; python tests/test_partition_example.py
  ```
  Lệnh này sẽ tự động:
  1. Kiểm tra gom cụm Spectral Clustering trên đồ thị.
  2. Tạo tệp cấu hình chỉ số `splits/spatial_skew.json` và `splits/temporal_shift.json`.
  3. Đo lường định lượng độ lệch phân phối bằng khoảng cách Wasserstein.
  4. Xuất 2 biểu đồ trực quan hóa tại `splits/spatial_skew_dist.png` và `splits/temporal_shift_dist.png`.

---

### 3.2. Chạy trên Linux / macOS (Bash)

```bash
# 1. FL Core test
PYTHONPATH=. python tests/test_fl_core.py

# 2. Data pipeline test
PYTHONPATH=. python tests/test_data_pipeline.py

# 3. Non-IID Partition tests
PYTHONPATH=. python tests/test_patrition.py
PYTHONPATH=. python tests/test_partition.py
```

Hoặc chạy toàn bộ test thông qua `pytest`:
```bash
PYTHONPATH=. pytest tests/
```

