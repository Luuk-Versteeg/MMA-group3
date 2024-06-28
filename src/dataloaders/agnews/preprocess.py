import numpy as np
import re


def preprocess(dataframe):
    dataframe["text"] = dataframe["text"].apply(clean_text)

    return dataframe


def clean_text(text):
    # Replace backticks
    text = text.replace("`", "")
    # Remove sequences of two or more dots
    text = re.sub(r'\.{2,}', '', text)
    return text
