
from pathlib import Path
import json

import numpy as np


class GraphTransformer:

    def __init__(
        self,
        input_path,
        mapping_path,
        output_dir,
        num_nodes=207,
    ):
        self.input_path = Path(input_path)
        self.mapping_path = Path(mapping_path)
        self.output_dir = Path(output_dir)

        self.num_nodes = num_nodes

        self.adj = None
        self.adj_sym = None
        self.sensor_ids = None

    def load(self):
        if not self.input_path.exists():
            raise FileNotFoundError(self.input_path)

        self.adj = np.load(self.input_path)

        if self.adj.shape != (
            self.num_nodes,
            self.num_nodes
        ):
            raise ValueError(
                f"Expected {(self.num_nodes, self.num_nodes)}, "
                f"got {self.adj.shape}"
            )

        self.adj = self.adj.astype(np.float32)

        return self

    def load_mapping(self):
        if not self.mapping_path.exists():
            raise FileNotFoundError(self.mapping_path)

        with open(
            self.mapping_path,
            "r",
            encoding="utf-8"
        ) as f:
            mapping = json.load(f)

        self.sensor_ids = mapping["sensor_ids"]

        if len(self.sensor_ids) != self.num_nodes:
            raise ValueError("Mapping không đủ sensor.")

        if len(set(self.sensor_ids)) != self.num_nodes:
            raise ValueError("Sensor ID bị trùng.")

        if mapping.get("shape") != [
            self.num_nodes,
            self.num_nodes
        ]:
            raise ValueError("Shape trong mapping không khớp.")

        return self

    def validate(self):
        if not np.isfinite(self.adj).all():
            raise ValueError("Adjacency chứa NaN hoặc Inf.")

        if np.any(self.adj < 0):
            raise ValueError("Adjacency chứa giá trị âm.")

        if np.any(self.adj > 1):
            raise ValueError("Adjacency vượt quá 1.")

        if not np.allclose(np.diag(self.adj), 1.0):
            raise ValueError("Đường chéo không bằng 1.")

        return self

    def symmetrize(self):
        """
        Ma trận gốc đã chứa Gaussian weights.
        Chỉ chuyển thành ma trận đối xứng.
        """

        self.adj_sym = np.maximum(
            self.adj,
            self.adj.T
        ).astype(np.float32)

        return self

    def validate_result(self):
        if not np.allclose(
            self.adj_sym,
            self.adj_sym.T,
            atol=1e-7
        ):
            raise ValueError("Adjacency chưa đối xứng.")

        if not np.isfinite(self.adj_sym).all():
            raise ValueError("Adjacency sau transform không hợp lệ.")

        if self.adj_sym.shape != (
            self.num_nodes,
            self.num_nodes
        ):
            raise ValueError("Sai kích thước adjacency.")

        return self

    def save(self):
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            self.output_dir / "adjacency_matrix.npy",
            self.adj_sym
        )

        with open(
            self.output_dir / "sensor_ids.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.sensor_ids,
                f,
                indent=4
            )

        print("Graph saved.")
        print("Adjacency:", self.adj_sym.shape)
        print("Sensors:", len(self.sensor_ids))

        return self

    def get_adjacency(self):
        return self.adj_sym

    def run(self):
        (
            self
            .load()
            .load_mapping()
            .validate()
            .symmetrize()
            .validate_result()
            .save()
        )

        return self
