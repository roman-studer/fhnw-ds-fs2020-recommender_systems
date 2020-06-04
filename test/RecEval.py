# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas, Roman
"""
from recommender.RS import RS
from data.DA import DA

from scipy.sparse import csr_matrix, csc_matrix, issparse
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import pandas as pd
import json


class RecEval(object):
    """
    Used to construct recommender and make evaluation possible
    """

    def __init__(self):
        rs = RS()
        da = DA.get_DA()  # maybe not necessary
        self.rs = rs
        self.da = da

        mae, nmae, rmse, precision_recall, = 'mae', 'nmae', 'rmse', 'precision_recall'
        self._evaluation_method = {mae: RecEval.mae,
                                   nmae: RecEval.nmae,
                                   rmse: RecEval.rmse,
                                   precision_recall: RecEval.precision_recall}

    def evaluate(self, mode, method, sim, recommender, nr_of_items, n=2, k=20, threshold=0.1, n_user=20):
        """
        Construct recommender and run evaluate function on it
        :param output: if True, prints output of evaluation method
        :param k: number of items to recommend
        :param n: number of items to mask
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :param method: selects the method to reduce the DataFrame (see product description)
        :param sim: defines the similarity method to be used
        :param recommender: defines if similarity matrix is for user-user or item-item matrix
        :param nr_of_items: number of items to display
        :param eval_method: method to evaluate recommender
        :param n_user: number of users to calculate prediction and recall with
        :return: returns output of evaluation function
        """
        # reduce data
        df = self.da.get_df_sub(method=method)

        self.rs.get_interaction(method=method, mode=mode, recommender=recommender)

        # get test_interaction
        test_interaction = self.rs.get_test_interaction(mode=mode, method=method, recommender=recommender)

        # find n last products per user
        last_products = self.get_last_n_products(df=df, n=n, method=method, mode=mode, recommender=recommender)

        # mask ratings for n predefined items
        last_products, masked_interaction = self.mask_item_rating(table=last_products, df=test_interaction)

        # predict
        predictions = self.rs.predict(R=masked_interaction, nr_of_items=nr_of_items, mode=mode, method=method, sim=sim)

        # get prediction rating to last_products
        last_products = self.get_prediction_rating(table=last_products, predictions=predictions)

        # run prediction accuracy
        mae = self._evaluation_method['mae'](last_products, own=True)
        nmae = self._evaluation_method['nmae'](last_products, own=True)
        rmse = self._evaluation_method['rmse'](last_products, own=True)
        precision, recall = self._evaluation_method['precision_recall'](self,
                                                                        method=method,
                                                                        recommender=recommender,
                                                                        last_products=last_products,
                                                                        P=predictions,
                                                                        k=k,
                                                                        mode=mode,
                                                                        threshold=0.1,
                                                                        n_user=n_user)

        evaluation = {'mae': mae, 'nmae': nmae, 'rmse': rmse, 'precision': precision, 'recall': recall}

        return evaluation

    @staticmethod
    def get_last_n_products(df, n, method, mode, recommender):
        """
        Gets n last products per user in dataframe and returns a list of these products
        :param df: Dataframe containing transaction records
        :param n: number of products to get, set to 2 by default
        :type df: pandas dataframe
        :type n: int
        """
        df = df.sort_values(['user_id', 'order_id'], ascending=[True, False])  # sort orders

        df = df.groupby(by='user_id').head(n).reset_index(drop=True)  # group by user and take n most recent product
        df = df[['user_id',
                 'order_id',
                 'product_name']]

        # load json
        inv_products = json.load(
            open(f'../data/interaction/products/{method}_{mode}_{recommender}_products.json', 'rb'))
        inv_users = json.load(open(f'../data/interaction/test/{method}_{mode}_{recommender}_test_users.json', 'rb'))

        products = {v: k for k, v in inv_products.items()}  # flip dictionary to get access to keys
        users = {v: k for k, v in inv_users.items()}  # flip dictionary to get access to keys

        product_id = []
        user_id = []

        # filter out transactions from unknown users:
        df = df[df['user_id'].isin(users.keys())]

        for row in df.iterrows():  # get corresponding rows and columns for user-item interaction
            p_id = products[row[1]['product_name']]
            user_val = users[row[1]['user_id']]
            product_id.append(p_id)
            user_id.append(user_val)

        df['row_id'] = product_id
        df['col_id'] = user_id
        return df[['user_id',
                   'product_name',
                   'row_id',
                   'col_id']]

    @staticmethod
    def mask_item_rating(table, df):
        """
        Masks n items per user and runs prediction on masked interaction matrix
        :param table: table with last n products per user
        :param df: interaction matrix
        :return: return df with columns 'user_id, product_name, rating, prediction' and masked interaction matrix
        :type df: sparse matrix
        """

        # convert R to sparse column matrix if not already done
        if isinstance(df, pd.DataFrame):
            df = csc_matrix(df.values)
        elif isinstance(df, np.matrix) or isinstance(df, np.ndarray):
            df = csc_matrix(df)
        elif issparse(df):
            pass

        rating = []
        for row in table.iterrows():
            user, product = row[1]['col_id'], row[1]['row_id']
            rating_p = df[int(user), int(product)]
            df[int(user), int(product)] = 0  # mask rating
            rating.append(rating_p)
        table['rating'] = rating

        return table, df

    @staticmethod
    def get_prediction_rating(table, predictions):
        """
        :param table: pandas df with last n products per user
        :param predictions: prediction matrix
        :return table: table with column 'rating' (predicted ratings)
        """
        # get ratings from prediction
        prediction = []
        predictions = csr_matrix(predictions)  # coo matrix doesn't support slicing, thus the change

        for row in table.iterrows():  # get rating per interaction
            user, product = row[1]['col_id'], row[1]['row_id']
            prediction_p = predictions[int(user), int(product)]
            prediction.append(prediction_p)

        table['prediction'] = prediction

        return table

    @staticmethod
    def mae(df, own=True):
        """
        Evaluation Metric: Calculate mean absolut error for two columns
        Formula taken from "Collaborative Filtering Recommender Systems by Michael D. Ekstrand et al
        Equation number: 3.1

        :param own: (bool) use own method ore if False method from sklearn
        :param output: (bool) print return value
        :param df: pandas dataframe containing ratings from evaluate function
        :return val: returns mean absolut error
        """
        if own:
            df['mae'] = df['prediction'] - df['rating']
            val = df['mae'].abs().sum() / len(df)
        else:
            val = mean_absolute_error(df['rating'], df['prediction'])

        return val

    @staticmethod
    def nmae(df, own=True):
        """
        Evaluation Metric: Calculate normalized mean absolut error for two columns
        Formula taken from "Collaborative Filtering Recommender Systems by Michael D. Ekstrand et al
        Equation number: 3.2

        :param own: (bool) use own method ore if False method from sklearn
        :param output: (bool) print return value
        :param df: pandas dataframe containing ratings from evaluate function
        :return val: returns mean absolut error
        """
        df['nmae'] = df['prediction'] - df['rating']
        val = df['nmae'].abs().sum() / (len(df) * (max(df['rating']) - min(df['rating'])))

        return val

    @staticmethod
    def rmse(df, own=True):
        """
        Evaluation Metric: Calculate root mean squared error for two columns
        Formula taken from "Collaborative Filtering Recommender Systems by Michael D. Ekstrand et al
        Equation number: 3.3

        :param own: (bool) use own method ore if False method from sklearn
        :param output: (bool) print return value
        :param df: pandas dataframe containing ratings from evaluate function
        :return val: returns mean absolut error
        """
        if own:

            df['rmse'] = (df['prediction'] - df['rating']) ** 2
            val = np.sqrt(df['rmse'].abs().sum() / len(df))

        else:
            val = mean_absolute_error(df['rating'], df['prediction'], squared=False)  # if squared True: Return is MSE

        return val

    # todo add filter for 'relevant' items in rec_item and masked_item
    def _precision_recall(self, user, user_number, last_products, P, k, mode, threshold=0.1):
        """"
        Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
        :param threshold: threshold to define a 'relevant' item
        :param last_products:
        :param user: int equal to user_id
        :param mode: mode of rating generation (binary, count, rating)
        :param k: number of items to recommend
        :param P: Prediction Matrix
        """
        # prediction matrix to csr
        if mode != 'binary':
            P = P.tocsr()

        # get k items to recommend
        rec_item, rec_prediction = self.rs.recommend_n(n=k, P=P, user=user_number, mode=mode)

        # get products of user
        df = last_products[last_products.user_id == user]
        masked_item = df['row_id'].tolist()

        masked_rating = df.rating.tolist()
        print(f'masked item: {masked_item}')
        print(f'rec item: {rec_item}')
        # precision
        precision = len(list(set(rec_item) & set(masked_item))) / k

        # recall
        try:
            recall = len(list(set(rec_item) & set(masked_item))) / len(
                [i for i in rec_prediction.tolist()[0] if i >= threshold])
        except:
            recall = 0
        return precision, recall

    def precision_recall(self, method, mode, recommender, last_products, P, k, threshold=0.1, n_user=20):
        sum_precision = 0
        sum_recall = 0
        users = last_products.user_id.unique()[:n_user]

        # load json
        inv_users = json.load(open(f'../data/interaction/test/{method}_{mode}_{recommender}_test_users.json', 'rb'))
        inv_users = {v: k for k, v in inv_users.items()}

        for user in users:
            user_number = int(inv_users.get(user))
            precision, recall = self._precision_recall(user=user,
                                                       user_number=user_number,
                                                       last_products=last_products,
                                                       P=P,
                                                       k=k,
                                                       mode=mode,
                                                       threshold=threshold)
            sum_precision += precision
            sum_recall += recall
        print(f'sum precision: {sum_precision}')
        print(f'sum recall: {sum_recall}')
        t_precision = sum_precision / len(users)
        t_recall = sum_recall / len(users)

        return t_precision, t_recall


if __name__ == '__main__':
    Rec = RecEval()
    mode, method, sim, recommender, nr_of_items, eval_method, n, threshold, k = ('count',
                                                                                 'freq',
                                                                                 'cosine',
                                                                                 'item',
                                                                                 20,
                                                                                 'mae',
                                                                                 5,
                                                                                 0.1,
                                                                                 20)
    Rec.evaluate(mode, method, sim, recommender, nr_of_items, n=2, k=20, threshold=0.1)
