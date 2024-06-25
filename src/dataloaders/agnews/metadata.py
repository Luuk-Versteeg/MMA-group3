from dash import html

name = "AG News"
scheme = "World, Sports, Business and Sci/Tech"
description = [
    html.Span([
        "AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining (clustering, classification, etc), information retrieval (ranking, search, etc), xml, data compression, data streaming, and any other non-commercial activity. For more information, please refer to this ", 
        html.A("link", href="http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html", target="_blank"),
        "."
        ]),
    html.P(["Hugging Face download link: ", html.A("AG News", href="https://huggingface.co/datasets/fancyzhx/ag_news", target="_blank")],)
]