import os
import shutil
from huggingface_hub import hf_hub_download

# Tạo thư mục lưu trữ cục bộ
save_dir = "data/raw/METR-LA"
os.makedirs(save_dir, exist_ok=True)

repo_id = "witgaw/METR-LA"
files = ["train.parquet", "val.parquet", "test.parquet"]

for f in files:
    # Lấy đường dẫn file đã tải trong cache và copy sang data/raw
    cached_path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
    dest_path = os.path.join(save_dir, f)
    shutil.copyfile(cached_path, dest_path)
    print(f"Đã đưa về project: {dest_path}")