
from pathlib import Path

from .graph_transformer import GraphTransformer
from .dataset_transformer import DatasetTransformer


class TransformPipeline:

    def __init__(
        self,
        raw_dir="data/raw/METR-LA",
        output_dir="data/processed/METR-LA",
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)

        self.graph_raw_dir = self.raw_dir / "sensor_graph"

    def validate_paths(self):
        """Kiểm tra các file đầu vào."""

        required_files = [
            self.raw_dir / "train.parquet",
            self.raw_dir / "test.parquet",
            self.graph_raw_dir / "adj_mx.npy",
            self.graph_raw_dir / "adj_mx_mapping.json",
        ]

        # Hỗ trợ val.parquet hoặc validation.parquet.
        val_candidates = [
            self.raw_dir / "val.parquet",
            self.raw_dir / "validation.parquet",
        ]

        if not any(path.exists() for path in val_candidates):
            raise FileNotFoundError(
                "Không tìm thấy val.parquet hoặc validation.parquet"
            )

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file: {path}"
                )

        print("All input paths validated.")

        return self

    def transform_graph(self):
        transformer = GraphTransformer(
            input_path=self.graph_raw_dir / "adj_mx.npy",
            mapping_path=self.graph_raw_dir / "adj_mx_mapping.json",
            output_dir=self.output_dir / "graph",
            num_nodes=207,
        )

        transformer.run()

        return transformer

    def transform_traffic(self):
        transformer = DatasetTransformer(
            raw_dir=self.raw_dir,
            output_dir=self.output_dir,
            num_nodes=207,
        )

        transformer.run()

        return transformer

    def run(self):
        print("=" * 60)
        print("METR-LA TRANSFORM PIPELINE")
        print("=" * 60)

        self.validate_paths()

        print("\n[1/2] Transform graph...")
        graph = self.transform_graph()

        print("\n[2/2] Transform traffic...")
        traffic = self.transform_traffic()

        print("\nTransform pipeline completed.")

        return {
            "graph": graph,
            "traffic": traffic,
        }
