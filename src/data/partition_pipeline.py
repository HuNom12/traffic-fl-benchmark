
from pathlib import Path

from .partition import generate_partitions


class PartitionPipeline:
    """
    Non-IID partition pipeline for METR-LA.

    Responsibilities:
    1. Load transformed traffic data and adjacency matrix.
    2. Generate spatial_skew partition.
    3. Generate temporal_shift partition.
    4. Save partition configurations.
    """

    def __init__(
        self,
        num_clients: int = 5,
        seed: int = 42,
    ):
        self.num_clients = num_clients
        self.seed = seed

        self.project_dir = Path(__file__).resolve().parents[2]

        self.processed_dir = (
            self.project_dir
            / "data"
            / "processed"
            / "METR-LA"
        )

        self.train_x_path = (
            self.processed_dir
            / "train"
            / "X_train.npy"
        )

        self.adjacency_path = (
            self.processed_dir
            / "graph"
            / "adjacency_matrix.npy"
        )

        self.splits_dir = self.project_dir / "splits"

    def validate(self):
        """Validate required transformed data."""

        print("\n========== PARTITION VALIDATION ==========")

        required_files = [
            self.train_x_path,
            self.adjacency_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

            print(f"[OK] Found: {path.relative_to(self.project_dir)}")

        self.splits_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self

    def partition(self):
        """Generate all Non-IID partition scenarios."""

        print("\n========== NON-IID PARTITION ==========")

        print(f"Number of clients: {self.num_clients}")
        print(f"Random seed: {self.seed}")

        generate_partitions(
            num_clients=self.num_clients,
            seed=self.seed,
        )

        print("\n[OK] Non-IID partitions generated.")

        return self

    def verify(self):
        """Verify generated partition files."""

        print("\n========== PARTITION VERIFICATION ==========")

        output_files = [
            self.splits_dir / "spatial_skew.json",
            self.splits_dir / "temporal_shift.json",
        ]

        for path in output_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Partition output not found: {path}"
                )

            print(f"[OK] Generated: {path.name}")

        return self

    def run(self):
        """Execute complete partition pipeline."""

        self.validate()
        self.partition()
        self.verify()

        print("\n========== PARTITION PIPELINE COMPLETED ==========")

        return self
