import os
import urllib.request
import pickle
import numpy as np

os.makedirs("data/raw/METR-LA", exist_ok=True)

# Link raw trực tiếp file đồ thị chuẩn của METR-LA từ repo chính thức
url = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl"
output_path = "data/raw/METR-LA/adj_mx.pkl"

print(f"Đang tải file ma trận kề từ: {url} ...")
try:
    urllib.request.urlretrieve(url, output_path)
    print(f"-> Đã tải thành công và lưu tại: {output_path}")

    # Đọc thử kiểm tra nội dung
    with open(output_path, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    print(f"-> Số lượng sensors: {len(sensor_ids)}")
    print(f"-> Shape ma trận kề A: {adj_mx.shape}")
    print(f"-> Kiểm tra giá trị: Min={np.min(adj_mx):.4f}, Max={np.max(adj_mx):.4f}")

except Exception as e:
    print(f"Lỗi khi tải hoặc đọc file: {e}")