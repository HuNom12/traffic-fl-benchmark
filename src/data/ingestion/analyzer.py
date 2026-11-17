
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd


class TrafficAnalyzer:
    """Analyze METR-LA traffic data and sensor graph metadata."""

    def __init__(self, raw_dir: str = "../../data/raw/METR-LA"):
        self.raw_dir = Path(raw_dir)
        self.graph_dir = self.raw_dir / "sensor_graph"

        self.splits = {}
        self.distances = None
        self.locations = None
        self.adjacency = None
        self.mapping = None

    # =========================================================
    # 1. LOAD RAW DATA
    # =========================================================

    def load_traffic_data(self):
        """Load available train, validation and test Parquet files."""

        split_names = ["train", "validation", "test"]

        for split in split_names:
            file_path = self.raw_dir / f"{split}.parquet"

            if not file_path.exists():
                print(f"[WARNING] Missing file: {file_path}")
                continue

            self.splits[split] = pd.read_parquet(file_path)

            print(
                f"[LOADED] {split}: "
                f"{self.splits[split].shape}"
            )

        return self

    def load_graph_data(self):
        """Load sensor graph files."""

        distance_path = self.graph_dir / "distances.csv"
        location_path = self.graph_dir / "sensor_locations.csv"
        adjacency_path = self.graph_dir / "adj_mx.npy"
        mapping_path = self.graph_dir / "adj_mx_mapping.json"

        if distance_path.exists():
            self.distances = pd.read_csv(distance_path)

        if location_path.exists():
            self.locations = pd.read_csv(location_path)

        if adjacency_path.exists():
            self.adjacency = np.load(adjacency_path)

        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)

        return self

    # =========================================================
    # 2. TRAFFIC DATA ANALYSIS
    # =========================================================

    def analyze_dataset(self):
        """Analyze all available traffic splits."""

        if not self.splits:
            print("[WARNING] No traffic data loaded.")
            return self

        print("\n" + "=" * 60)
        print("TRAFFIC DATA ANALYSIS")
        print("=" * 60)

        for split_name, df in self.splits.items():

            print(f"\n--- {split_name.upper()} ---")

            print(f"Shape: {df.shape}")
            print(f"Number of columns: {len(df.columns)}")

            print("\nColumn names:")
            print(df.columns.tolist())

            print("\nData types:")
            print(df.dtypes)

            # Missing values
            missing = df.isna().sum()
            missing = missing[missing > 0]

            print("\nMissing values:")
            if missing.empty:
                print("No missing values.")
            else:
                print(missing.sort_values(ascending=False))

            # Duplicate rows
            duplicate_count = df.duplicated().sum()

            print(f"\nDuplicate rows: {duplicate_count:,}")

            # Numeric statistics
            print("\nNumeric statistics:")
            print(df.describe().T)

            # Check infinity
            numeric_df = df.select_dtypes(include=np.number)

            if not numeric_df.empty:
                inf_count = np.isinf(
                    numeric_df.to_numpy(dtype=np.float64)
                ).sum()

                print(f"Infinite values: {inf_count:,}")

            # Inspect X and Y columns
            x_columns = [
                col for col in df.columns
                if str(col).startswith("x_")
            ]

            y_columns = [
                col for col in df.columns
                if str(col).startswith("y_")
            ]

            print(f"\nInput X columns: {len(x_columns)}")
            print(f"Target Y columns: {len(y_columns)}")

            print("X sample:", x_columns[:5])
            print("Y sample:", y_columns[:5])

            if not x_columns:
                print("[WARNING] No columns starting with x_.")

            if not y_columns:
                print("[WARNING] No columns starting with y_.")

        return self

    # =========================================================
    # 3. SENSOR DISTANCE ANALYSIS
    # =========================================================

    def analyze_distances(self):
        """Analyze pairwise sensor distances."""

        print("\n" + "=" * 60)
        print("SENSOR DISTANCE ANALYSIS")
        print("=" * 60)

        if self.distances is None:
            print("[WARNING] distances.csv not found.")
            return self

        df = self.distances

        print("\nShape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nPreview:")
        print(df.head())

        required_columns = {"from", "to", "cost"}

        if not required_columns.issubset(df.columns):
            print(
                "[ERROR] Expected columns: "
                "from, to, cost"
            )
            return self

        print("\nMissing values:")
        print(df[["from", "to", "cost"]].isna().sum())

        print("\nDuplicate edges:", df.duplicated(
            subset=["from", "to"]
        ).sum())

        print("\nNumber of unique source sensors:",
              df["from"].nunique())

        print("Number of unique target sensors:",
              df["to"].nunique())

        print("\nDistance statistics (meters):")
        print(df["cost"].describe())

        # Invalid distances
        invalid = (
            df["cost"].isna()
            | ~np.isfinite(df["cost"])
            | (df["cost"] < 0)
        )

        print("\nInvalid distance records:", invalid.sum())

        # Sensor IDs
        sensor_ids = set(df["from"].dropna()) | set(
            df["to"].dropna()
        )

        print("Total unique sensors in distance file:",
              len(sensor_ids))

        return self

    # =========================================================
    # 4. SENSOR LOCATION ANALYSIS
    # =========================================================

    def analyze_locations(self):
        """Analyze sensor geographic metadata."""

        print("\n" + "=" * 60)
        print("SENSOR LOCATION ANALYSIS")
        print("=" * 60)

        if self.locations is None:
            print("[WARNING] sensor_locations.csv not found.")
            return self

        df = self.locations

        print("\nShape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("\nPreview:")
        print(df.head())

        print("\nMissing values:")
        print(df.isna().sum())

        print("\nDuplicate rows:", df.duplicated().sum())

        # Identify possible coordinate columns
        lower_columns = {
            str(col).lower(): col for col in df.columns
        }

        lat_col = next(
            (
                lower_columns[name]
                for name in ["latitude", "lat"]
                if name in lower_columns
            ),
            None
        )

        lon_col = next(
            (
                lower_columns[name]
                for name in ["longitude", "lon", "lng"]
                if name in lower_columns
            ),
            None
        )

        if lat_col is None or lon_col is None:
            print(
                "[WARNING] Cannot identify latitude/longitude "
                "columns automatically."
            )
            return self

        print("\nLatitude statistics:")
        print(df[lat_col].describe())

        print("\nLongitude statistics:")
        print(df[lon_col].describe())

        invalid_coordinates = (
            ~df[lat_col].between(-90, 90)
            | ~df[lon_col].between(-180, 180)
        )

        print(
            "\nInvalid coordinates:",
            invalid_coordinates.sum()
        )

        return self

    # =========================================================
    # 5. ADJACENCY MATRIX ANALYSIS
    # =========================================================

    def analyze_adjacency(self):
        """Analyze precomputed adjacency matrix."""

        print("\n" + "=" * 60)
        print("ADJACENCY MATRIX ANALYSIS")
        print("=" * 60)

        if self.adjacency is None:
            print("[WARNING] adj_mx.npy not found.")
            return self

        adj = self.adjacency

        print("\nShape:", adj.shape)
        print("Data type:", adj.dtype)

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            print("[ERROR] Adjacency matrix must be square.")
            return self

        print("Minimum:", np.nanmin(adj))
        print("Maximum:", np.nanmax(adj))

        print("NaN values:", np.isnan(adj).sum())
        print("Infinite values:", np.isinf(adj).sum())

        print(
            "Symmetric:",
            np.allclose(adj, adj.T, equal_nan=True)
        )

        print(
            "Diagonal equals 1:",
            np.allclose(np.diag(adj), 1)
        )

        nonzero = np.count_nonzero(adj)
        total = adj.size

        print(f"Non-zero entries: {nonzero:,}/{total:,}")
        print(f"Density: {nonzero / total:.4%}")

        return self

    # =========================================================
    # 6. SENSOR MAPPING ANALYSIS
    # =========================================================

    def analyze_mapping(self):
        """Inspect sensor-to-node mapping metadata."""

        print("\n" + "=" * 60)
        print("SENSOR MAPPING ANALYSIS")
        print("=" * 60)

        if self.mapping is None:
            print("[WARNING] adj_mx_mapping.json not found.")
            return self

        print("\nMapping type:", type(self.mapping).__name__)

        if isinstance(self.mapping, dict):
            print("Mapping keys:", list(self.mapping.keys()))

            print("\nMapping preview:")
            print(json.dumps(
                self.mapping,
                indent=2,
                ensure_ascii=False
            )[:3000])

        elif isinstance(self.mapping, list):
            print("Mapping length:", len(self.mapping))
            print("Mapping preview:", self.mapping[:10])

        return self

    # =========================================================
    # 7. CROSS-DATA VALIDATION
    # =========================================================

    def validate_sensor_consistency(self):
        """Compare sensor counts across traffic and graph files."""

        print("\n" + "=" * 60)
        print("SENSOR CONSISTENCY CHECK")
        print("=" * 60)

        distance_ids = set()

        if self.distances is not None:
            if {"from", "to"}.issubset(self.distances.columns):
                distance_ids = (
                    set(self.distances["from"].dropna())
                    | set(self.distances["to"].dropna())
                )

        location_ids = set()

        if self.locations is not None:
            id_col = next(
                (
                    col for col in self.locations.columns
                    if str(col).lower() in [
                        "sensor_id", "station", "node_id", "id"
                    ]
                ),
                None
            )

            if id_col is not None:
                location_ids = set(
                    self.locations[id_col].dropna()
                )

        print("Sensors in distances:", len(distance_ids))
        print("Sensors in locations:", len(location_ids))

        if self.adjacency is not None:
            print(
                "Sensors in adjacency:",
                self.adjacency.shape[0]
            )

        if distance_ids and location_ids:
            print(
                "Distance IDs missing in locations:",
                len(distance_ids - location_ids)
            )

            print(
                "Location IDs missing in distances:",
                len(location_ids - distance_ids)
            )

        if self.adjacency is not None and distance_ids:
            if self.adjacency.shape[0] != len(distance_ids):
                print(
                    "[WARNING] Adjacency size does not match "
                    "number of distance sensors."
                )

        return self

    # =========================================================
    # 8. RUN ALL ANALYSIS
    # =========================================================

    def analyze_all(self):
        """Run all traffic and graph analysis."""

        self.load_traffic_data()
        self.load_graph_data()

        self.analyze_dataset()
        self.analyze_distances()
        self.analyze_locations()
        self.analyze_adjacency()
        self.analyze_mapping()
        self.validate_sensor_consistency()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED")
        print("=" * 60)

        return self
