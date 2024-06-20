import pandas as pd
from pathlib import Path

POLARITY_DOWNLOAD_URL = "hf://datasets/fancyzhx/amazon_polarity/amazon_polarity/"

DATA_FOLDER = str(Path(__file__).parent) + "/data/"

TRAIN_DATA_PATH = "train.parquet"
TRAIN_DATA_PATH1 = "train-00000-of-00004.parquet"
TRAIN_DATA_PATH2 = "train-00001-of-00004.parquet"
TRAIN_DATA_PATH3 = "train-00002-of-00004.parquet"
TRAIN_DATA_PATH4 = "train-00003-of-00004.parquet"
TEST_DATA_PATH = "test-00000-of-00001.parquet"

labels = {
    0: "Negative",
    1: "Positive",
}

try:
    polarity_train = pd.read_parquet(DATA_FOLDER + TRAIN_DATA_PATH)
    polarity_test = pd.read_parquet(DATA_FOLDER + TEST_DATA_PATH)

except FileNotFoundError as e:
    print("Downloading POLARITY dataset...")
    polarity_train1 = pd.read_parquet(POLARITY_DOWNLOAD_URL + TRAIN_DATA_PATH1)
    polarity_train2 = pd.read_parquet(POLARITY_DOWNLOAD_URL + TRAIN_DATA_PATH2)
    polarity_train3 = pd.read_parquet(POLARITY_DOWNLOAD_URL + TRAIN_DATA_PATH3)
    polarity_train4 = pd.read_parquet(POLARITY_DOWNLOAD_URL + TRAIN_DATA_PATH4)

    polarity_train = pd.concat([polarity_train1, polarity_train2, polarity_train3, polarity_train4])
    polarity_test = pd.read_parquet(POLARITY_DOWNLOAD_URL + TEST_DATA_PATH)

    polarity_train["label"] = polarity_train["label"].map(labels)
    polarity_test["label"] = polarity_test["label"].map(labels)

    polarity_train = polarity_train.rename(columns={"content": "text"})
    polarity_test = polarity_test.rename(columns={"content": "text"})

    polarity_train = polarity_train.drop(columns=["title"])
    polarity_test = polarity_test.drop(columns=["title"])

    print("Saving POLARITY dataset...")
    polarity_train.to_parquet(DATA_FOLDER + TRAIN_DATA_PATH)
    polarity_test.to_parquet(DATA_FOLDER + TEST_DATA_PATH)
