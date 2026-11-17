from src.data.ingestion.bronze_pipeline import BronzePipeline

pipeline = BronzePipeline(
    raw_dir="data/raw/METR-LA"
)

pipeline.run()