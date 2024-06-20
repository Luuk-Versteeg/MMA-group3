import pandas as pd

import dataloaders.agnews.metadata as agnews
from dataloaders.agnews.download import agnews_train, agnews_test, labels as ag_labels

import dataloaders.amazon_polarity.metadata as amazon

DATALOADER_PATH = "src/dataloaders/"

datasets = [
    {
        "name": agnews.name,
        "scheme": agnews.scheme,
        "description": agnews.description,
        "data": {
            "train": agnews_train,
            "test": agnews_test,
        },
        "labels": ag_labels,
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