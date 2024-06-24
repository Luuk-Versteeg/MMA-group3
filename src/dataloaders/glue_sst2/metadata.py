from dash import html


name = "GLUE/sst2"
scheme = "Negative, Positive"
description = [
    "GLUE, the ",
    html.A("General Language Understanding Evaluation benchmark", href="https://gluebenchmark.com/", target="_blank"), 
    " is a collection of resources for training, evaluating, and analyzing natural language understanding systems. The Stanford Sentiment Treebank (sst2) consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.",
    html.P(["Hugging Face download link: ", html.A("GLUE", href="https://huggingface.co/datasets/nyu-mll/glue", target="_blank")],)
]