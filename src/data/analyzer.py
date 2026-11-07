
import pandas as pd


class TrafficAnalyzer:
    """
    Analyze station-level traffic dataset.

    Traffic variables:
    - total_flow
    - avg_occupancy
    - avg_speed
    """

    TRAFFIC_COLUMNS = [
        "total_flow",
        "avg_occupancy",
        "avg_speed"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        self.required_columns = [
            "timestamp",
            "station",
            *self.TRAFFIC_COLUMNS
        ]

        self._validate_columns()

        # Extract date from timestamp
        self.df["date"] = pd.to_datetime(
            self.df["timestamp"]
        ).dt.date

    # ==========================================
    # 1. VALIDATE DATA
    # ==========================================

    def _validate_columns(self):
        missing = [
            col for col in self.required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    # ==========================================
    # 2. GENERAL DATA STATISTICS
    # ==========================================

    def general_statistics(self):
        """
        Statistics of the entire dataset.
        """

        print("\n========== GENERAL STATISTICS ==========")

        print(f"Total records: {len(self.df):,}")

        print(
            f"Total stations: "
            f"{self.df['station'].nunique():,}"
        )

        print(
            f"Total days: "
            f"{self.df['date'].nunique():,}"
        )

        print(
            f"Time range: "
            f"{self.df['timestamp'].min()} "
            f"-> {self.df['timestamp'].max()}"
        )

        print("\nMissing values:")

        missing_stats = pd.DataFrame({
            "missing_count": self.df.isna().sum(),
            "missing_percent": (
                self.df.isna().mean() * 100
            ).round(2)
        })

        print(missing_stats)

        print("\nTraffic statistics:")

        print(
            self.df[self.TRAFFIC_COLUMNS].describe().T
        )

        return self

    # ==========================================
    # 3. STATIONS WITHOUT ANY TRAFFIC
    # ==========================================

    def stations_without_any_traffic(self):
        """
        Count stations that have all 3 traffic
        variables missing throughout a day.
        """

        traffic_missing = self.df[
            self.TRAFFIC_COLUMNS
        ].isna().all(axis=1)

        daily_stats = (
            self.df.assign(no_traffic=traffic_missing)
            .groupby("date")
            .apply(
                lambda group: group.loc[
                    group["no_traffic"], "station"
                ].nunique(),
                include_groups=False
            )
        )

        print("\n========== NO TRAFFIC ==========")

        print(daily_stats.rename(
            "stations_without_any_traffic"
        ))

        return daily_stats

    # ==========================================
    # 4. STATIONS WITHOUT SPEED AND OCCUPANCY
    # ==========================================

    def stations_without_speed_occupancy(self):
        """
        Count stations that have both avg_speed
        and avg_occupancy missing throughout a day.
        """

        condition = self.df[
            ["avg_speed", "avg_occupancy"]
        ].isna().all(axis=1)

        daily_stats = (
            self.df.assign(missing_both=condition)
            .groupby("date")
            .apply(
                lambda group: group.loc[
                    group["missing_both"], "station"
                ].nunique(),
                include_groups=False
            )
        )

        print("\n========== NO SPEED AND OCCUPANCY ==========")

        print(daily_stats.rename(
            "stations_without_speed_occupancy"
        ))

        return daily_stats

    # ==========================================
    # 5. STATIONS WITHOUT SPEED
    # ==========================================

    def stations_without_speed(self):
        """
        Count stations that have avg_speed
        missing throughout a day.
        """

        condition = self.df["avg_speed"].isna()

        daily_stats = (
            self.df.assign(missing_speed=condition)
            .groupby("date")
            .apply(
                lambda group: group.loc[
                    group["missing_speed"], "station"
                ].nunique(),
                include_groups=False
            )
        )

        print("\n========== NO SPEED ==========")

        print(daily_stats.rename(
            "stations_without_speed"
        ))

        return daily_stats

    def stations_without_occupancy(self):
        """
        Count stations that have avg_occupancy
        missing throughout a day.
        """

        condition = self.df["avg_occupancy"].isna()

        daily_stats = (
            self.df.assign(missing_occupancy=condition)
            .groupby("date")
            .apply(
                lambda group: group.loc[
                    group["missing_occupancy"], "station"
                ].nunique(),
                include_groups=False
            )
        )

        print("\n========== NO OCCUPANCY ==========")

        print(daily_stats.rename(
            "stations_without_occupancy"
        ))

        return daily_stats

    def stations_without_flow(self):
        """
        Count stations that have total_flow
        missing throughout a day.
        """

        condition = self.df["total_flow"].isna()

        daily_stats = (
            self.df.assign(missing_flow=condition)
            .groupby("date")
            .apply(
                lambda group: group.loc[
                    group["missing_flow"], "station"
                ].nunique(),
                include_groups=False
            )
        )

        print("\n========== NO FLOW ==========")

        print(daily_stats.rename(
            "stations_without_flow"
        ))

        return daily_stats