import pandas as pd

import dataloaders.agnews.metadata as agnews
import dataloaders.amazon_polarity.metadata as amazon

DATALOADER_PATH = "src/dataloaders/"

datasets = [
    {
        "name": agnews.name,
        "scheme": agnews.scheme,
        "description": agnews.description,
        "data": {
            "train": pd.read_csv(DATALOADER_PATH + "agnews/data/train.csv"),
            "test": pd.read_csv(DATALOADER_PATH + "agnews/data/test.csv")
        }
    },
    {
        "name": amazon.name,
        "scheme": amazon.scheme,
        "description": amazon.description,
        "data": {
            "test": pd.read_parquet(DATALOADER_PATH + "amazon_polarity/data/test-00000-of-00001.parquet", engine="pyarrow"),
            # TODO add more splits
        }
    }
]