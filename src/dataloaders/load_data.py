import dataloaders.agnews.metadata as agnews
from dataloaders.agnews.loader import agnews_train, agnews_test, labels as ag_labels
from dataloaders.agnews.preprocess import preprocess as preprocess_agnews

import dataloaders.amazon_polarity.metadata as polarity
from dataloaders.amazon_polarity.loader import polarity_train, polarity_test, labels as polarity_labels
from dataloaders.amazon_polarity.preprocess import preprocess as preprocess_polarity

import dataloaders.glue_sst2.metadata as glue
from dataloaders.glue_sst2.loader import glue_train, glue_validation, labels as glue_labels
from dataloaders.glue_sst2.preprocess import preprocess as preprocess_glue


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
        "preprocess": preprocess_agnews,
    },
    {
        "name": polarity.name,
        "scheme": polarity.scheme,
        "description": polarity.description,
        "data": {
            "train": polarity_train,
            "test": polarity_test,
        },
        "labels": polarity_labels,
        "preprocess": preprocess_polarity
    },
    {
        "name": glue.name,
        "scheme": glue.scheme,
        "description": glue.description,
        "data": {
            "train": glue_train,
            "validation": glue_validation,
        },
        "labels": glue_labels,
        "preprocess": preprocess_glue
    }
]