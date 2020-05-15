import pandas as pd
from pathlib import Path

chunk = pd.read_csv('Recommender4Retail_big.csv', nrows=10_00_000)
chunk = chunk.loc[:,['user_id', 'order_id', 'product_name', 'product_id', 'reordered', 'aisle_id']]
print(chunk.head())
if not Path('Recommender4Retail.csv').is_file():
    chunk.to_csv('Recommender4Retail.csv', mode='a', header=True, index=True)
