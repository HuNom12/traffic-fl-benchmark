import os

# Danh sách thư mục cần tạo
directories = [
    "configs/dataset",
    "configs/non_iid",
    "data/raw",
    "data/processed",
    "splits",
    "src/data",
    "src/models",
    "src/federated",
    "src/utils",
    "tests",
    "checkpoints",
    "logs",
]

# Danh sách file rỗng cần khởi tạo sẵn
files = [
    ".gitignore",
    "requirements.txt",
    "README.md",
    "configs/config.yaml",
    "configs/dataset/pems04.yaml",
    "configs/non_iid/spatial_skew.yaml",
    "configs/non_iid/temporal_shift.yaml",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "splits/.gitkeep",
    "checkpoints/.gitkeep",
    "logs/.gitkeep",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/partition.py",
    "src/models/__init__.py",
    "src/models/base.py",
    "src/models/temporal.py",
    "src/models/spatio_temporal.py",
    "src/federated/__init__.py",
    "src/federated/client.py",
    "src/federated/server.py",
    "src/federated/trainer.py",
    "src/utils/__init__.py",
    "src/utils/metrics.py",
    "src/utils/seed.py",
    "tests/__init__.py",
    "tests/test_pipeline.py",
]

for d in directories:
    os.makedirs(d, exist_ok=True)

for f in files:
    if not os.path.exists(f):
        with open(f, "w", encoding="utf-8") as fp:
            pass

print("Dựng khung dự án hoàn tất!")