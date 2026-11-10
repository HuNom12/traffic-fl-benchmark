
import pandas as pd
from pathlib import Path
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class TrafficTransformer:
    """
    Transform traffic dataset from lane-level to station-level.

    Responsibilities:
    - Remove lane-related attributes.
    - Select station-level features.
    - Sort data by station and timestamp.
    - Provide an extensible structure for future transformations.
    """
    GROUP_COLUMNS = [
        "freeway",
        "direction_of_travel",
        "lane_type"
    ]

    TRAFFIC_COLUMNS = [
        "total_flow",
        "avg_occupancy",
        "avg_speed"
    ]
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.original_shape = df.shape

        self.selected_columns = [
            "timestamp",
            "station",
            "district",
            "freeway",
            "direction_of_travel",
            "lane_type",
            "station_length",
            "samples",
            "percent_observed",
            "total_flow",
            "avg_occupancy",
            "avg_speed"
        ]

    # ==========================================
    # 1. SELECT STATION-LEVEL FEATURES
    # ==========================================

    def remove_lane_features(self):
        """
        Remove all lane-related attributes.
        Keep station-level traffic information.
        """

        missing_columns = [
            col for col in self.selected_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        self.df = self.df[self.selected_columns].copy()

        print(
            f"Lane features removed: "
            f"{self.original_shape[1]} -> {self.df.shape[1]} columns"
        )

        return self

    # ==========================================
    # 2. SORT DATA
    # ==========================================

    def sort_data(self):
        """
        Sort dataset by station and timestamp.
        """

        self.df = self.df.sort_values(
            by=["station", "timestamp"]
        ).reset_index(drop=True)

        return self

    # ==========================================
    # 3. CHECK DATA
    # ==========================================

    def get_info(self):
        """
        Display dataset information.
        """

        print("Dataset shape:", self.df.shape)
        print("\nColumns:")
        print(self.df.columns.tolist())

        print("\nData information:")
        self.df.info()

        return self

    # ==========================================
    # 4. GET RESULT
    # ==========================================

    def get_data(self):
        """
        Return transformed DataFrame.
        """

        return self.df

    # ==========================================
    # 5. SAVE DATA
    # ==========================================

    def save(self, output_path):
        """
        Save transformed dataset.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df.to_parquet(
            output_path,
            index=False
        )

        print(f"Dataset saved to: {output_path}")

        return self


 
    def remove_stations_without_traffic(self):
        """
        Remove stations that have absolutely no valid
        flow, occupancy, or speed measurements.

        A station is removed only when all 3 traffic
        columns are NaN across all its records.
        """

        traffic_columns = [
            "total_flow",
            "avg_occupancy",
            "avg_speed"
        ]

        # Validate columns
        missing_columns = [
            col for col in traffic_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        before_stations = self.df["station"].nunique()
        before_rows = len(self.df)

        # Check whether each station has at least
        # one valid traffic value across all timestamps.
        station_has_traffic = (
            self.df
            .groupby("station")[traffic_columns]
            .apply(
                lambda group: group.notna().any().any()
            )
        )

        # Stations with no valid traffic information
        invalid_stations = station_has_traffic[
            ~station_has_traffic
        ].index

        # Remove all records belonging to invalid stations
        self.df = self.df[
            ~self.df["station"].isin(invalid_stations)
        ].copy()

        after_stations = self.df["station"].nunique()
        after_rows = len(self.df)

        print("\n========== REMOVE EMPTY STATIONS ==========")
        print(f"Stations before: {before_stations}")
        print(f"Stations removed: {len(invalid_stations)}")
        print(f"Stations after: {after_stations}")

        print(f"Rows before: {before_rows:,}")
        print(f"Rows removed: {before_rows - after_rows:,}")
        print(f"Rows after: {after_rows:,}")

        print("\nRemoved station IDs:")
        print(invalid_stations.tolist())

        return self

    def impute_flow(self):
        """
        Impute missing values in total_flow.

        Strategy:
        1. Linear interpolation within each station.
        - Only fill gaps of at most 1 consecutive timestamp.
        - Do not extrapolate at the beginning or end.
        2. Fill remaining missing values using station median.
        3. Track original missing values and imputed values.
        """

        self.df = self.df.copy()

        # Sắp xếp theo station và timestamp
        self.df = self.df.sort_values(
            ["station", "timestamp"]
        ).reset_index(drop=True)

        # Đánh dấu các giá trị missing ban đầu
        self.df["flow_was_missing"] = (
            self.df["total_flow"].isna()
        )

        # Bước 1: Nội suy tuyến tính trong từng station
        self.df["total_flow"] = (
            self.df.groupby("station")["total_flow"]
            .transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit=1,
                    limit_area="inside"
                )
            )
        )

        # Bước 2: Tính median của từng station
        station_median = (
            self.df.groupby("station")["total_flow"]
            .transform("median")
        )

        # Fallback bằng station median
        self.df["total_flow"] = (
            self.df["total_flow"].fillna(station_median)
        )

        # Đánh dấu các giá trị đã được impute
        self.df["flow_is_imputed"] = (
            self.df["flow_was_missing"]
            & self.df["total_flow"].notna()
        )

        # Thống kê kết quả
        original_missing = self.df["flow_was_missing"].sum()
        remaining_missing = self.df["total_flow"].isna().sum()
        imputed = self.df["flow_is_imputed"].sum()

        print("===== FLOW IMPUTATION =====")
        print(f"Original missing : {original_missing}")
        print(f"Remaining missing: {remaining_missing}")
        print(f"Imputed          : {imputed}")

        return self

    def flow_based_imputation(self):
        """
        Metadata + Flow-based Imputation.

        Sử dụng total_flow, metadata và thời gian
        để dự đoán avg_speed và avg_occupancy.

        Chỉ dự đoán khi total_flow có dữ liệu.
        Không ghi đè các giá trị traffic gốc.
        """

        target_columns = [
            "avg_speed",
            "avg_occupancy"
        ]

        categorical_features = [
            "freeway",
            "direction_of_travel",
            "lane_type",
            "district"
        ]

        required_columns = [
            "timestamp",
            "total_flow"
        ] + target_columns

        missing_columns = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        # Tạo bản sao để không ảnh hưởng dữ liệu gốc
        df = self.df.copy().reset_index(drop=True)

        # Chuyển timestamp về datetime
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        # Chuyển traffic features về numeric
        numeric_columns = [
            "total_flow",
            "avg_speed",
            "avg_occupancy"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Feature engineering từ timestamp
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["dayofweek"] = df["timestamp"].dt.dayofweek

        # Mã hóa chu kỳ trong ngày
        minute_of_day = (
            df["hour"] * 60 + df["minute"]
        )

        df["time_sin"] = np.sin(
            2 * np.pi * minute_of_day / 1440
        )

        df["time_cos"] = np.cos(
            2 * np.pi * minute_of_day / 1440
        )

        # Chỉ sử dụng metadata thực sự tồn tại
        categorical_features = [
            col for col in categorical_features
            if col in df.columns
        ]

        numeric_features = [
            "total_flow",
            "hour",
            "minute",
            "dayofweek",
            "time_sin",
            "time_cos"
        ]

        feature_columns = (
            numeric_features + categorical_features
        )

        # Chuẩn bị metadata
        for col in categorical_features:
            df[col] = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
            )

        # Tạo preprocessing pipeline
        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    numeric_features
                ),
                (
                    "cat",
                    categorical_transformer,
                    categorical_features
                )
            ]
        )

        # Báo cáo missing trước xử lý
        missing_before = df[target_columns].isna().sum()

        print("\nFlow-based Imputation")
        print("Missing before:")
        print(missing_before)

        for target in target_columns:

            # Chỉ dùng những dòng có target và Flow hợp lệ
            train_mask = (
                df[target].notna()
                & df["total_flow"].notna()
                & df["timestamp"].notna()
            )

            # Chỉ dự đoán những dòng target bị thiếu
            # nhưng Flow và timestamp còn dữ liệu
            predict_mask = (
                df[target].isna()
                & df["total_flow"].notna()
                & df["timestamp"].notna()
            )

            train_count = int(train_mask.sum())
            predict_count = int(predict_mask.sum())

            print(f"\nTarget: {target}")
            print(f"Training samples: {train_count}")
            print(f"Missing samples to predict: {predict_count}")

            if predict_count == 0:
                print("No missing values eligible for prediction.")
                continue

            if train_count < 20:
                print(
                    "Not enough training samples. "
                    "Skip this target."
                )
                continue

            # X và y
            X_train = df.loc[
                train_mask, feature_columns
            ]

            y_train = df.loc[
                train_mask, target
            ]

            X_missing = df.loc[
                predict_mask, feature_columns
            ]

            # Random Forest model
            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                ]
            )

            # Training
            model.fit(X_train, y_train)

            # Prediction
            predictions = model.predict(X_missing)

            # Chỉ cập nhật những vị trí missing
            target_values = df[target].to_numpy(copy=True)

            target_values[predict_mask.to_numpy()] = predictions

            df[target] = target_values

            print(f"Successfully predicted: {len(predictions)}")

        # Báo cáo sau xử lý
        missing_after = df[target_columns].isna().sum()

        report = pd.DataFrame({
            "missing_before": missing_before,
            "missing_after": missing_after,
            "filled": missing_before - missing_after
        })

        print("\nFinal Imputation Report")
        print(report)

        # Cập nhật DataFrame của Transformer
        self.df = df.drop(
            columns=[
                "hour",
                "minute",
                "dayofweek",
                "time_sin",
                "time_cos"
            ]
        )

        return self


