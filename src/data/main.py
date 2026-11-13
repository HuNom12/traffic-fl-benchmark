from src.data.ingestion.loader import DataLoader
import pandas as pd

'''Download METR-LA dataset from Hugging Face and save it as raw Parquet files.
loader = DataLoader(
    data_dir="../../data/raw"
)'''


train_df = pd.read_parquet("../data/raw/METR-LA/train.parquet")
val_df = pd.read_parquet("data/raw/METR-LA/val.parquet")
test_df = pd.read_parquet("data/raw/METR-LA/test.parquet")

print(train_df)
print(val_df)
print(test_df)

