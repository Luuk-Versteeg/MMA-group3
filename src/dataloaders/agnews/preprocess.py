import numpy as np
import re


def preprocess(dataframe):

    # Discards all rows containing these charaters.
    discard = ["#", "&"]

    dataframe = dataframe[~dataframe.text.str.contains('|'.join(discard))]

    # Replace certain characters with spaces
    dataframe["text"] = np.where(
        dataframe["text"].str.contains(r'\\'),
        dataframe["text"].str.replace("\\", " "), 
        dataframe["text"]
    )

    dataframe["text"] = dataframe["text"].apply(clean_text)

    return dataframe


def clean_text(text):
    # Replace backticks
    text = text.replace("`", "")
    # Remove sequences of two or more dots
    text = re.sub(r'\.{2,}', '', text)
    return text
