traffic-fl-benchmark/
├── configs/                      # Cấu hình tham số hệ thống (YAML)
│   ├── config.yaml               # Cấu hình trung tâm (toàn cục)
│   ├── dataset/                  # Cấu hình dữ liệu (metr_la.yaml, pems04.yaml [rỗng])
│   └── non_iid/                  # Cấu hình kịch bản Non-IID (spatial_skew, temporal_shift)
├── data/                         # Lưu trữ dữ liệu các tầng
│   ├── raw/                      # Bronze: Dữ liệu thô tải từ HuggingFace & metadata đồ thị
│   └── processed/                # Silver: Tensor 4D đã qua tiền xử lý, sliding window & scaler
├── splits/                       # Gold: Chỉ số phân hoạch Non-IID cho từng Client (.json)
├── src/                          # Mã nguồn cốt lõi (Core Engine)
│   ├── data/                     # Module xử lý dữ liệu
│   │   ├── ingestion/            # Tải và kiểm tra tính toàn vẹn dữ liệu thô (Bronze)
│   │   ├── transform/            # Trích xuất đặc trưng, Z-score, Tensor 4D hóa (Silver)
│   │   ├── dataset.py            # PyTorch Dataset (StandardTrafficDataset) & DataLoader factory
│   │   ├── partition.py          # Thuật toán phân hoạch Non-IID (Spatial Skew / Temporal Shift)
│   │   ├── partition_pipeline.py # Điều phối sinh partition
│   │   └── data_pipeline.py      # Pipeline dữ liệu chạy chuỗi: Bronze -> Transform -> Partition
│   ├── models/                   # Module mô hình dự báo
│   │   ├── base.py               # Lớp trừu tượng BaseForecaster (chuẩn hóa giao diện tensor 4D)
│   │   ├── temporal.py           # Baseline thuần thời gian (GRUForecaster, LSTMForecaster)
│   │   ├── spatio_temporal.py    # Mô hình không-thời gian STGCNForecaster (Chebyshev / GCN)
│   │   └── __init__.py           # Model Registry & Factory function `build_model`
│   ├── federated/                # Hạ tầng Federated Learning
│   │   ├── client.py             # ClientNode (hỗ trợ FedAvg & FedProx)
│   │   ├── server.py             # FederatedServer (Tổng hợp trọng số mô hình)
│   │   └── trainer.py            # FederatedTrainer (Điều phối Centralized, Local, Federated)
│   └── utils/                    # Tiện ích bổ trợ
│       ├── seed.py               # Khóa seed tái lập thực nghiệm (Reproducibility)
│       └── metrics.py            # Khử chuẩn hóa (Denormalization) & Đánh giá (MAE, RMSE, MAPE)
├── tests/                        # Bộ kiểm thử đơn vị và tích hợp (Unit & Integration Tests)
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md                     # Tài liệu hướng dẫn dự án