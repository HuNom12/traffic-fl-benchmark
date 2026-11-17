
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
import pandas as pd
from datasets import load_dataset, DatasetDict


class DataLoader:
    """Load and save traffic datasets."""

    def __init__(self, data_dir: str = "../../data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_metr_la_from_huggingface(self) -> DatasetDict:
        """
        Download METR-LA dataset from Hugging Face.

        Returns:
            DatasetDict containing available splits.
        """
        print("Downloading METR-LA from Hugging Face...")

        ds = load_dataset("witgaw/METR-LA")

        print("\nDataset downloaded successfully!")
        print(f"Available splits: {list(ds.keys())}")

        for split_name, split_data in ds.items():
            print(
                f"{split_name}: "
                f"{len(split_data):,} records, "
                f"{len(split_data.column_names)} columns"
            )

        return ds

    def save_metr_la_raw(
        self,
        ds: DatasetDict,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Save each available split as a raw Parquet file.
        """
        save_dir = (
            Path(output_dir)
            if output_dir
            else self.data_dir / "METR-LA"
        )

        save_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in ds.items():
            output_path = save_dir / f"{split_name}.parquet"

            split_data.to_parquet(str(output_path))

            print(
                f"Saved {split_name}: {output_path} "
                f"({len(split_data):,} records)"
            )

        print("\nMETR-LA raw data saved successfully!")

    def download_metr_la(self) -> DatasetDict:
        """
        Download METR-LA and save all splits into data/raw/METR-LA.
        """
        ds = self.load_metr_la_from_huggingface()
        self.save_metr_la_raw(ds)

        return ds
    
    def download_metr_la_graph(self):
        """
        Download METR-LA sensor graph metadata.
        """
        output_dir = self.data_dir / "METR-LA"

        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id="witgaw/METR-LA",
            repo_type="dataset",
            allow_patterns="sensor_graph/*",
            local_dir=str(output_dir)
        )

        print(f"Graph data saved at: {output_dir}")

    def download_all_metr_la(self):
        """
        Download complete METR-LA raw dataset:
        1. Traffic data
        2. Sensor graph metadata
        """

        print("=" * 60)
        print("START METR-LA RAW INGESTION")
        print("=" * 60)

        # Download traffic splits
        print("\n[1/2] Downloading traffic data...")
        self.download_metr_la()

        # Download graph metadata
        print("\n[2/2] Downloading sensor graph...")
        self.download_metr_la_graph()

        print("\n" + "=" * 60)
        print("METR-LA RAW INGESTION COMPLETED")
        print("=" * 60)

    def load(self, file_name: str) -> pd.DataFrame:
        """
        Load a local Parquet file.
        """
        file_path = self.data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {file_path}"
            )

        return pd.read_parquet(file_path)
