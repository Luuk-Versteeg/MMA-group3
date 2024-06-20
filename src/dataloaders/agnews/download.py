import pandas as pd
from pathlib import Path

AGNEWS_DOWNLOAD_URL = "hf://datasets/fancyzhx/ag_news/data/"
TRAIN_DATA_PATH = "train-00000-of-00001.parquet"
TEST_DATA_PATH = "test-00000-of-00001.parquet"

DATA_FOLDER = str(Path(__file__).parent) + "/data/"

labels = {
    0: "World", 
    1: "Sports", 
    2: "Business", 
    3: "Sci/Tech"
}

try:
    agnews_train = pd.read_parquet(DATA_FOLDER + TRAIN_DATA_PATH)
    agnews_test = pd.read_parquet(DATA_FOLDER + TEST_DATA_PATH)

except FileNotFoundError as e:
    agnews_train = pd.read_parquet(AGNEWS_DOWNLOAD_URL + TRAIN_DATA_PATH)
    agnews_test = pd.read_parquet(AGNEWS_DOWNLOAD_URL + TEST_DATA_PATH)

    agnews_train["label"] = agnews_train["label"].map(labels)
    agnews_test["label"] = agnews_test["label"].map(labels)

    print("Saving AGNEWS dataset...")
    agnews_train.to_parquet(DATA_FOLDER + TRAIN_DATA_PATH)
    agnews_test.to_parquet(DATA_FOLDER + TEST_DATA_PATH)
