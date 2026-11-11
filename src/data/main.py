from src.data.ingestion.loader import DataLoader
from src.data.transform.graph_transformer import TrafficTransformer
from src.data.ingestion.analyzer import TrafficAnalyzer
from split import TrafficSplitter

'''Download METR-LA dataset from Hugging Face and save it as raw Parquet files.'''
loader = DataLoader(
    data_dir="../../data/raw"
)

ds = loader.download_metr_la()
loader.download_metr_la_graph()

'''train_df = pd.read_parquet("METR-LA/train.parquet")
val_df = pd.read_parquet("METR-LA/val.parquet")
test_df = pd.read_parquet("METR-LA/test.parquet")'''

