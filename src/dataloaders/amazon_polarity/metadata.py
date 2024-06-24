from dash import html

name = "Amazon Polarity"
scheme = "Negative, Positive"
description = [
    html.Span("The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review."),
    html.P(["Hugging Face download link: ", html.A("Amazon Polarity", href="https://huggingface.co/datasets/fancyzhx/amazon_polarity", target="_blank")])
]
