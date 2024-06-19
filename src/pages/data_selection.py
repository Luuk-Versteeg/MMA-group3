from datasets import load_dataset

glue_dataset = load_dataset("nyu-mll/glue", "sst2", split="validation")
amazon_dataset = load_dataset("fancyzhx/amazon_polarity", split="test")
ag_news_dataset = load_dataset("fancyzhx/ag_news", split="test")