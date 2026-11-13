from src.data.ingestion.bronze_pipeline import BronzePipeline
from src.data.partition_pipeline import PartitionPipeline
from src.data.transform.transform_pipeline import TransformPipeline

if __name__ == "__main__":
    pipeline0 = BronzePipeline(
        raw_dir="data/raw/METR-LA"
    )

    pipeline0.run()

    pipeline1 = TransformPipeline(
        raw_dir="data/raw/METR-LA",
        output_dir="data/processed/METR-LA",
    )

    pipeline1.run()

    pipeline2 = PartitionPipeline(
        num_clients=5,
        seed=42,
    )

    pipeline2.run()