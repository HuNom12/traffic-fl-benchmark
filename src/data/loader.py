
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class TrafficLoader:
    """
    Load traffic data from CSV, Parquet and TXT files.
    """

    SUPPORTED_FORMATS = {".csv", ".parquet", ".txt"}

    def __init__(self, data_dir: Optional[str] = None):
        """
        Args:
            data_dir: Root directory containing raw datasets.
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()  

    def load(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Automatically detect file format and load data.

        Args:
            file_path: Path to input file.
            **kwargs: Additional arguments for pandas readers.

        Returns:
            pd.DataFrame
        """

        path = Path(file_path)

        # Resolve relative paths from data_dir
        if not path.is_absolute():
            path = self.data_dir / path

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {extension}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        try:
            if extension == ".csv":
                return self._load_csv(path, **kwargs)

            elif extension == ".parquet":
                return self._load_parquet(path, **kwargs)

            elif extension == ".txt":
                return self._load_txt(path, **kwargs)

        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataset {path}: {error}"
            ) from error

    @staticmethod
    def _load_csv(
        path: Path,
        **kwargs
    ) -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def _load_parquet(
        path: Path,
        **kwargs
    ) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(path, **kwargs)

    @staticmethod
    def _load_txt(
        path: Path,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load TXT file.

        Default delimiter is comma.
        Override with sep='\\t' for tab-separated files.
        """
        return pd.read_csv(
            path,
            sep=kwargs.pop("sep", ","),
            **kwargs
        )

    def load_multiple(
        self,
        file_paths: list,
        **kwargs
    ) -> dict:
        """
        Load multiple datasets.

        Returns:
            Dictionary: {filename: DataFrame}
        """
        datasets = {}

        for file_path in file_paths:
            path = Path(file_path)

            datasets[path.stem] = self.load(
                file_path,
                **kwargs
            )

        return datasets

    def save(self, output_path, file_format="parquet"):
        """
        Save transformed DataFrame.

        Supported formats:
        - parquet
        - csv
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if file_format == "parquet":
            if output_path.suffix != ".parquet":
                output_path = output_path.with_suffix(".parquet")

            self.df.to_parquet(
                output_path,
                index=False
            )

        elif file_format == "csv":
            if output_path.suffix != ".csv":
                output_path = output_path.with_suffix(".csv")

            self.df.to_csv(
                output_path,
                index=False
            )

        else:
            raise ValueError(
                "Chỉ hỗ trợ parquet và csv."
            )

        print(f"Saved transformed data: {output_path}")
        print(f"Shape: {self.df.shape}")

        return self