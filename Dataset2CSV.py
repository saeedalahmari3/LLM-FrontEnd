import pandas as pd

df = pd.read_parquet("hf://datasets/MBZUAI/LaMini-Hallucination/data/test-00000-of-00001-57a3961ffd6c22d8.parquet")

df.to_csv('../mbzuai.csv')
