import pandas as pd
from pathlib import Path

GLUE_DOWNLOAD_URL = "hf://datasets/nyu-mll/glue/sst2/"
TRAIN_DATA_PATH = "train-00000-of-00001.parquet"
VALIDATION_DATA_PATH = "validation-00000-of-00001.parquet"
# TEST_DATA_PATH = "test-00000-of-00001.parquet"

# NOTE: Cannot use test dataset because it has only one label (-1) instead of 0 and 1

DATA_FOLDER = str(Path(__file__).parent) + "/data/"

labels = {
    0: "Negative",
    1: "Positive"
}

try:
    glue_train = pd.read_parquet(DATA_FOLDER + TRAIN_DATA_PATH)
    glue_validation = pd.read_parquet(DATA_FOLDER + VALIDATION_DATA_PATH)
    # glue_test = pd.read_parquet(DATA_FOLDER + TEST_DATA_PATH)

except FileNotFoundError as e:
    print("Downloading GLUE/sst2 dataset...")
    glue_train = pd.read_parquet(GLUE_DOWNLOAD_URL + TRAIN_DATA_PATH)
    glue_validation = pd.read_parquet(GLUE_DOWNLOAD_URL + VALIDATION_DATA_PATH)
    # glue_test = pd.read_parquet(GLUE_DOWNLOAD_URL + TEST_DATA_PATH)

    glue_train["label"] = glue_train["label"].map(labels)
    glue_validation["label"] = glue_validation["label"].map(labels)
    # glue_test["label"] = glue_test["label"].map(labels)

    glue_train = glue_train.rename(columns={"sentence": "text"})
    glue_validation = glue_validation.rename(columns={"sentence": "text"})
    # glue_test = glue_test.rename(columns={"sentence": "text"})

    glue_train = glue_train.drop(columns=["idx"])
    glue_validation = glue_validation.drop(columns=["idx"])
    # glue_test = glue_test.drop(columns=["idx"])

    print("Saving GLUE/sst2 dataset...")
    glue_train.to_parquet(DATA_FOLDER + TRAIN_DATA_PATH)
    glue_validation.to_parquet(DATA_FOLDER + VALIDATION_DATA_PATH)
    # glue_test.to_parquet(DATA_FOLDER + TEST_DATA_PATH)
