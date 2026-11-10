from loader import TrafficLoader
from transform import TrafficTransformer
from analyzer import TrafficAnalyzer
from split import TrafficSplitter

loader = TrafficLoader()

df = loader.load("../../data/raw/01/d11_text_station_5min_2026_01_01.parquet")
print(df.head())
transformer = TrafficTransformer(df)

transformed_df = (
    transformer
    .remove_lane_features()
    .sort_data()
    .remove_stations_without_traffic()
    .impute_flow()
    .flow_based_imputation()
    .save(
        "../../data/processed/transformed/traffic_transformed1.parquet"
    )
    .get_data()

)

print(transformed_df.head())


splitter = TrafficSplitter(
    data=transformed_df,
    seq_len=12,
    pred_len=12,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2
)

result = splitter.run()

# Lấy dữ liệu
train_x = result["train_x"]
train_y = result["train_y"]

val_x = result["val_x"]
val_y = result["val_y"]

test_x = result["test_x"]
test_y = result["test_y"]

mean = result["mean"]
std = result["std"]

# Initialize analyzer
analyzer = TrafficAnalyzer(transformed_df)

# Run analysis
analyzer.general_statistics()

no_traffic = analyzer.stations_without_any_traffic()

no_speed_occupancy = (
    analyzer.stations_without_speed_occupancy()
)

no_speed = analyzer.stations_without_speed()

no_occupancy = analyzer.stations_without_occupancy()

no_flow = analyzer.stations_without_flow()