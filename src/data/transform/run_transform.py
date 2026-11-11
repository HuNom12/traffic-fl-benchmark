
from src.data.transform.transform_pipeline import TransformPipeline


if __name__ == "__main__":
    pipeline = TransformPipeline(
        raw_dir="data/raw/METR-LA",
        output_dir="data/processed/METR-LA",
    )

    pipeline.run()
