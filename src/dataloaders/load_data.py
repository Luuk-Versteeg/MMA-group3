import dataloaders.agnews.metadata as agnews
from dataloaders.agnews.loader import agnews_train, agnews_test, labels as ag_labels

import dataloaders.amazon_polarity.metadata as polarity
from dataloaders.amazon_polarity.loader import polarity_train, polarity_test, labels as polarity_labels


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
        "name": polarity.name,
        "scheme": polarity.scheme,
        "description": polarity.description,
        "data": {
            "train": polarity_train,
            "test": polarity_test,
        },
        "labels": polarity_labels
    }
]