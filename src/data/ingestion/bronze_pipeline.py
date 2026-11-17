from pathlib import Path

from .loader import DataLoader
from .analyzer import TrafficAnalyzer


class BronzePipeline:
    """
    Raw ingestion pipeline for METR-LA.

    Responsibilities:
    1. Download traffic data.
    2. Download sensor graph metadata.
    3. Verify raw data.
    """

    def __init__(
        self,
        raw_dir: str = "../../data/raw/METR-LA"
    ):
        self.raw_dir = Path(raw_dir)

        self.loader = DataLoader(
            data_dir=str(self.raw_dir.parent)
        )

        self.analyzer = TrafficAnalyzer(
            raw_dir=str(self.raw_dir)
        )

    def ingest(self):
        """Download all METR-LA raw data."""

        print("\n========== BRONZE INGESTION ==========")

        self.loader.download_all_metr_la()

        print("\n[OK] Raw data ingestion completed.")

        return self

    def validate(self):
        """Analyze downloaded raw data."""

        print("\n========== BRONZE VALIDATION ==========")

        self.analyzer.analyze_all()

        return self

    def run(self):
        """Execute complete ingestion pipeline."""

        self.ingest()
        self.validate()

        print("\n========== PIPELINE COMPLETED ==========")

        return self