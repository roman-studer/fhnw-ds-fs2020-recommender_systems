import pandas as pd
import numpy as np

def doc_get_transaction():
    data = np.array([[1,1,2,3,1,2,3,3,3,3],['A','A','B','B','C','D','C','C','C','D']]).T
    transaction = pd.DataFrame(data, columns = ['user_id','product_name'])
    return transaction

def doc_interaction_count(transaction):
    interaction_count = transaction.pivot_table(index='user_id', columns='product_name', aggfunc=len, fill_value=0)

    return interaction_count

def doc_interaction_unary(transaction):
    interaction_unary = transaction.pivot_table(index='user_id', columns='product_name', aggfunc=len, fill_value=0).applymap(lambda x: int(x>0))

    return interaction_unary
