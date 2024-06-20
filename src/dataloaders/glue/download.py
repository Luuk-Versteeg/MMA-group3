


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet("hf://datasets/nyu-mll/glue/ax/test-00000-of-00001.parquet")
    import pdb; pdb.set_trace()