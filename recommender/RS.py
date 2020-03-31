# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, csc_matrix
from pandas.api.types import CategoricalDtype

from recommender._Recommender_Init import _RecommenderInit
from data.DA import DA
import os


class RS(_RecommenderInit):
    """
    Example class of a recommender. Every Recommender will have its own class
    structured.
    """

    def __init__(self):
        super().__init__()  # maybe not necessary
        self.DA = DA.get_DA()

        # is self.data and self._param still necessary?
        '''self.data = data
        self._param = param  # maybe store some parameter when initializing object? idk'''

        binary_freq, binary_aisle, binary_rating, count_int_freq, count_int_aisle, count_int_rating = \
            'binary_freq', 'binary_asile', 'binary_rating', 'count_int_freq', 'count_int_aisle', 'count_int_rating'

        freq_binary_item, freq_count_item, freq_rating_item, aisle_binary_item, aisle_count_item, \
            aisle_rating_item, rating_binary_item, rating_count_item, rating_rating_item, \
            freq_binary_user, freq_count_user, freq_rating_user, aisle_binary_user, \
            aisle_count_user, aisle_rating_user, rating_binary_user, rating_count_user, rating_rating_user = \
            'freq_binary_item', 'freq_count_item', 'freq_rating_item', 'aisle_binary_item', 'aisle_count_item', \
            'aisle_rating_item', 'rating_binary_item', 'rating_count_item', 'rating_rating_item', \
            'freq_binary_user', 'freq_count_user', 'freq_rating_user', 'aisle_binary_user', \
            'aisle_count_user', 'aisle_rating_user', 'rating_binary_user', 'rating_count_user', 'rating_rating_user'

        self._interaction = {freq_binary_item: None, freq_count_item: None, freq_rating_item: None,
                             aisle_binary_item: None, aisle_count_item: None,
                             aisle_rating_item: None, rating_binary_item: None, rating_count_item: None,
                             rating_rating_item: None, freq_binary_user: None, freq_count_user: None,
                             freq_rating_user: None, aisle_binary_user: None, aisle_count_user: None,
                             aisle_rating_user: None, rating_binary_user: None,
                             rating_count_user: None, rating_rating_user: None}

        cosine, pearson = 'cosine', 'pearson'
        self._similarity_method = {cosine: self.sim_cosine, pearson: self.sim_pearson}

    def get_interaction(self, method='freq', mode='binary', recommender='item'):
        """
        Creates an interaction matrix with shape (users,products)
        :param method: selects the method to reduce the dataframe (see product description)
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :param recommender: defines if output is for a user-user matrix or a item-item matrix (csr or csc matrix)
        :return: numpy array
        """
        # check if interaction matrix already exists:
        path = self.DA._nav + method + '_' + mode + '_' + recommender + '_interaction.pkl'
        matrix = method + '_' + mode + '_' + recommender
        if os.path.exists(path):
            self._interaction[matrix] = pickle.load(open(path, "rb"))

        # create interaction_matrix
        else:
            # get data
            self.DA.get_df_sub(method)
            df = self.DA._df_sub_data[method]

            # create interaction matrix
            user_c = CategoricalDtype(sorted(df.user_id.unique()), ordered=True)
            product_name_c = CategoricalDtype(sorted(df.product_name.unique()), ordered=True)

            row = df.user_id.astype(user_c).cat.codes
            col = df.product_name.astype(product_name_c).cat.codes
            if mode == 'count' or mode == 'binary':
                val = [1] * len(col)
            elif mode == 'rating':
                val = df.rating
            else:
                raise AssertionError(f'Parameter mode needs to be str "cont", "binary" or "rating" not {mode}')

            # csr_matrix for user-user or csc_matrix for item-item
            if recommender == 'user':
                df = csr_matrix((val, (row, col)), shape=(user_c.categories.size, product_name_c.categories.size))
            elif recommender == 'item':
                df = csc_matrix((val, (row, col)), shape=(user_c.categories.size, product_name_c.categories.size))
            else:
                raise AssertionError(f'Parameter recommender needs to be str "user" or "item" not {recommender}')

            if mode == 'binary':
                df[df.nonzero()] = 1

            self._interaction[matrix] = df

            # save interaction matrix
            pickle.dump(df, open(path, "wb"))

        return self._interaction[matrix]

    def similarity(self, method, mode, interaction='count_int_freq', sim='cosine', recommender='item'):
        """
        Creates a similarity matrix with given function of shape (n,n)
        :param df: interaction matrix (sparse matrix)
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :param method: selects the method to reduce the dataframe (see product description)
        :param interaction: Interaction Matrix to be used
        :param sim: defines the similarity method to be used
        """
        # check if user-user or item-item matrix already exists

        # check if interaction matrix already exists:
        path = self.DA._nav + method + '_' + mode + '_' + recommender + '_similarity.pkl'
        matrix = interaction + '_' + sim
        if os.path.exists(path):
            self._interaction[matrix] = pickle.load(open(path, "rb"))
        else:
            # load correct df here
            df = self.get_df_interaction(method, mode, recommender)

            # create empty similarity_matrix(nxn)
            if recommender == 'item':
                length = df.shape[1]
                similarity_matrix = np.zeros((length, length))

                # get value pairs:
                for i in range(length):
                    a = np.asarray(df[:, i].todense()).T[0]
                    for j in range(length):
                        # fill empty similarity_matrix
                        b = np.asarray(df[:, j].todense()).T[0]
                        similarity_matrix[i, j] = self._similarity_method[sim](a, b)

            elif recommender == 'user':
                length = df.shape[0]
                similarity_matrix = np.zeros((length, length))

                # get value pairs:
                for i in range(length):
                    a = np.asarray(df[i, :].todense())[0]

                    for j in range(length):
                        # fill empty similarity_matrix
                        b = np.asarray(df[j, :].todense())[0]
                        similarity_matrix[i, j] = self._similarity_method[sim](a, b)

            pickle.dump(similarity_matrix, open(path, 'wb'))

        return similarity_matrix

    @staticmethod
    def sim_cosine(v1, v2, norm=False):
        """
        Calculates the cosine similarity between two given vectors
        :param v1: vector in a numpy array format
        :param v2: vector in a numpy array format
        :param norm: if norm is set to true the vectors will be normalized to unit vectors before the calculation
        :return s: returns a float value between 0 and 1
        """
        # check if params have correct type (comment this out later)
        if isinstance(v1, np.ndarray) is False or isinstance(v2, np.ndarray) is False:
            raise TypeError(f'Function only accepts v1 and v2 as type numpy.ndarray')

        # normalize vectors
        if norm:
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

        # cosine similarity:
        s = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        return s

    @staticmethod
    def sim_pearson(u, v, threshold=True):
        """
        Calculates the pearson similarity between two given users
        :param u: vector of all ratings for user/item
        :param v: vector of all ratings for user/item
        :param threshold: activate involvement of a threshold "multiplication" of the function
        :return s: similarity value between -1 and 1 (1 high correlation, 0 no correlation, -1 high negative correlation) or NaN if calculation is not possible
        """
        # get overlapping items for user u and v inclusive ratings.
        intersection = [k for k, i, j in zip(np.arange(len(u)), u, v) if i > 0 and j > 0]

        u_mean_r, v_mean_r = np.average(u, weights=(u > 0)), np.average(v, weights=(v > 0))  # average rating

        numerator = sum(
            a * b for a, b in zip([u[i] - u_mean_r for i in intersection], [v[i] - v_mean_r for i in intersection]))
        denominator1 = np.sqrt(sum([(u[i] - u_mean_r) ** 2 for i in intersection]))
        denominator2 = np.sqrt(sum([(v[i] - v_mean_r) ** 2 for i in intersection]))

        if numerator / (denominator1 * denominator2) == 0:
            s = np.nan
        else:
            s = numerator / (denominator1 * denominator2)
        if threshold:
            s = s * min(len(interesection) / 50, 1)

        return s

    def transform(self, data, return_type):
        # TO DO: correct placement of function?
        # used to transform data into a fitting format for the recommender
        if type(data) != return_type:
            if return_type == "dataframe":
                data = pd.DataFrame(data)
            else:
                data = data.values()
        return data

    def train_test(self, data):
        # TO DO: maybe move to _RecommenderInit
        # splits data into train_test if necessary
        return df_train, df_test

    def fit(self, df_train):
        return None

    def recommend(self, user):
        # user may be a list or object?

        return prediction

    def get_df_interaction(self, method, mode, recommender):
        """Lazy loader of the interaction matrix"""
        matrix = method + '_' + mode + '_' + recommender
        if not isinstance(self._interaction[matrix], (csc_matrix, csr_matrix)):
            path = self.DA._nav + matrix + '_interaction.pkl'
            if os.path.exists(path):
                self._interaction[matrix] = pickle.load(open(path, 'rb'))
            else:
                self._interaction[matrix] = self.get_interaction(method, mode, recommender)

        return self._interaction[matrix]


if __name__ == '__main__':
    B = RS()
    B.similarity(mode='count', method='rating', sim='cosine', recommender='item')

# old interaction function, new one uses a sparse matrix for better performance
'''    def get_interaction(self, mode='binary', method='freq', pivot=False):
        """
        Creates an interaction matrix with shape (users,products)
        :param method: selects the method to reduce the dataframe (see product description)
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :return: numpy array
        """
        # check if interaction matrix already exists:
        path = self._nav + method + '_interaction_' + mode + '.csv'
        matrix = mode + '_' + method
        if os.path.exists(path):
            self._interaction[matrix] = pd.read_csv(path)
        # create interaction_matrix
        else:
            # get data
            self.get_df_sub(method)
            df = self._df_sub_data[method]

            # create interaction matrix
            if pivot:
                if mode == 'count':
                    df = df.pivot_table(index='user_id', columns='product_name', aggfunc=len, fill_value=0)
                elif mode == 'binary':
                    df = df.pivot_table(index='user_id', columns='product_name', aggfunc=len, fill_value=0)

                    # helperfunction
                    def val_to_binary(x):
                        if x > 0:
                            x = 1
                        else:
                            x = 0
                        return x

                    df = df.applymap(val_to_binary)
                else:
                    raise Exception(
                        'Function "get_interaction" only accepts mode "binary" and "count" not "{}"'.format(mode))

            self._interaction[matrix] = df

            # save interaction matrix
            self._interaction[matrix].to_csv(path, index=False)
        return self._interaction[matrix].to_numpy()'''
