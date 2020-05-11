import pickle
import numpy as np
from scipy.sparse.linalg import norm

with open('freq_count_item_interaction.pkl','rb') as f:
    df = pickle.load(f)

#print(norm(df[:,5114]))
t =norm(df, axis=0)
print(t[5144])