import pickle
import numpy as np
with open('freq_rating_item_similarity.pkl', 'rb') as f:
    df = pickle.load(f)

print(df)