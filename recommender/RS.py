# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

from numpy.core._multiarray_umath import ndarray
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import norm
from pandas.api.types import CategoricalDtype
from recommender._Recommender_Init import _RecommenderInit
from data.DA import DA
import os
import warnings


class RS(_RecommenderInit):
    """
    Example class of a recommender. Every Recommender will have its own class
    structured.
    """

    def __init__(self):
        super().__init__()
        self._da = DA.get_DA()
        
        cosine, pearson, jaccard = 'cosine', 'pearson', 'jaccard'
        self._similarity_method = {cosine: RS.sim_cosine, pearson: RS.sim_pearson, jaccard: RS.sim_jaccard}
        
        self._interaction_matrix = None
        self._interaction_method, self._interaction_mode, self._interaction_recommender = None, None, None

    def _get_df_interaction(self, method, mode, recommender):
        """
        Creates an interaction matrix with shape (users,products)
        :param method: selects the method to reduce the dataframe (see product description)
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :param recommender: defines if output is for a user-user matrix or a item-item matrix (csr or csc matrix)
        :return: numpy array
        """
        # check if interaction matrix already exists:
        path = self._da.get_nav() + 'interaction/' + method + '_' + mode + '_' + recommender + '_interaction.pkl'
        if os.path.exists(path):
            interaction_matrix = pickle.load(open(path, "rb"))

        # create interaction_matrix
        else:
            df_sub = self._da.get_df_sub(method)
     
            #creates the number of orders per unique product
            df_o = df_sub.groupby(by = ['user_id', 'product_name']).count().rename(columns = {'order_id':'o'}).reset_index()
            #creates the total number of orders from a user
            o_tot = df_sub.groupby(by = ['user_id'])['order_id'].nunique()
            #creates a new dataframe where user_id, product_id, o, o_tot
            df_orders = df_o.join(o_tot, on = 'user_id').rename(columns = {'order_id':'o_tot'})
            #generates the rating for val
            o = df_orders.loc[:, 'o'].to_numpy()
            
            if mode == 'count':
                val = o
            elif mode == 'binary':
                val = [1] * len(o)
            elif mode == 'rating':
                o_tot = df_orders.loc[:, 'o_tot'].to_numpy()
                rating_fun = np.vectorize(self._rating)
                val = rating_fun(o = o, o_tot = o_tot).T
            else:
                raise AssertionError(f'Parameter mode needs to be str "cont", "binary" or "rating" not {mode}')
                    
            user_c = CategoricalDtype(sorted(df_orders.user_id.unique()), ordered=True)
            product_name_c = CategoricalDtype(sorted(df_orders.product_name.unique()), ordered=True)

            row = df_orders.user_id.astype(user_c).cat.codes
            col = df_orders.product_name.astype(product_name_c).cat.codes

            # csr_matrix for user-user or csc_matrix for item-item
            if recommender == 'user':
                interaction_matrix = csr_matrix((val, (row, col)), shape=(user_c.categories.size, product_name_c.categories.size))
            elif recommender == 'item':
                interaction_matrix = csc_matrix((val, (row, col)), shape=(user_c.categories.size, product_name_c.categories.size))
            else:
                raise AssertionError(f'Parameter recommender needs to be str "user" or "item" not {recommender}')

            # save interaction matrix
            pickle.dump(interaction_matrix, open(path, "wb"))

        return interaction_matrix
    
    def _rating(self, o, o_tot, m = 10, omega = 1/3):
        """
        Product rating for each user
        :param o: np.array of number of orders for each product
        :param o_tot: np.array of the total number of orders per user (must have same length as `o`)
        :param m: parameter of how strongly low amount of purchases counts in the rating (if m is high, then lower purchases counts less)
        :param omega: parameter how to weight the first purchse. The rest adjusts automatically that the result is always between 0 and 1
        
        Calculates a user-product rating considering number of purchases of a product and number of all purchases per user
        """
        if o == 0:
            x = 0
        elif o == 1:
            x = omega
        else:
            if o_tot < m:
                w_freq = np.sqrt(o_tot / m)
            else:
                w_freq = 1
            w_prod = np.sqrt(o / o_tot)
            x = omega + (1-omega) * w_prod * w_freq
        return x
    
    def get_interaction(self, method='freq', mode='binary', recommender='item'):
        """Lazy loader of the interaction matrix"""
        if isinstance(self._interaction_matrix, (csc_matrix, csr_matrix)) and\
        method == self._interaction_method and mode == self._interaction_mode and recommender == self._interaction_recommender:
            pass #nothing toDo here, correct matrix is already in self._interaction_matrix
        else:
            self._interaction_matrix = self._get_df_interaction(method, mode, recommender)
            self._interaction_method = method
            self._interaction_mode = mode
            self._interaction_recommender = recommender

        return self._interaction_matrix

    def product_names(self, method='freq'):
        """
        returns a list of products contained in the df. The index is the corresponding index number in the similarity matrix.
        """
        df = self._da.get_df_sub(method)
        product = pd.DataFrame(sorted(df.product_name.unique()))

        return product

    def similarity(self, method, mode, sim='cosine', recommender='item'):
        """
        Creates a similarity matrix with given function of shape (n,n)
        :param df: interaction matrix (sparse matrix)
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :param method: selects the method to reduce the dataframe (see product description)
        :param interaction: Interaction Matrix to be used
        :param sim: defines the similarity method to be used
        :param recommender: defines if similarity matrix is for user-user or item-item matrix
        :return similarity_matrix: nxn-Matrix containing the similarity values for every value pair
        """
        # check if user-user or item-item matrix already exists
        path_sim = self._da.get_nav() + 'similarity/' + method + '_' + mode + '_' + recommender + '_similarity.pkl'
        if os.path.exists(path_sim):
            similarity_matrix = pickle.load(open(path_sim, 'rb'))

        # check if interaction matrix already exists:
        else:
            df_interaction = self.get_interaction(method, mode, recommender)
            similarity_matrix = self._similarity_method[sim](df_interaction, recommender)
            pickle.dump(similarity_matrix, open(path_sim, 'wb'))

        return similarity_matrix

    @staticmethod
    def sim_cosine(df, recommender):
        """
        Calculates the cosine similarity between two given vectors
        :param time: displays the estimated time for execution if set to True
        :param recommender: differenctiates between ubcf or ibcf recommender
        :param df: sparse matrix with shape (user,item)
        :return s: similarity value between -1 and 1 (1 high correlation, 0 no correlation, -1 high negative correlation)
        """

        if recommender == 'item':

            # initialize empty diagonal matrix
            length = df.shape[1]
            similarity_matrix = np.zeros((length, length), dtype=np.float32)  # empty similarity matrix

            # precalculate normalized vector over whole df
            normalized_vectors = norm(df, axis=0)

            for i in np.arange(length):
                # cosine similarity calculation
                a = df[:, i]
                numerator = df.T.dot(a).todense().T  # get the dotproduct for vector a and every other vector
                denominator = normalized_vectors * normalized_vectors[i]

                s = numerator / denominator
                similarity_matrix[i, :] = s


        elif recommender == 'user':

            # initialize empty diagonal matrix
            length = df.shape[0]
            similarity_matrix = np.zeros((length, length), dtype=np.float32)  # empty similarity matrix

            # precalculate normalized vector over whole df
            normalized_vectors = norm(df, axis=1)

            for i in np.arange(length):
                # cosine similarity calculation
                a = df[i, :]
                numerator = df.dot(a.T).todense().T  # get the dotproduct for vector a and every other vector
                denominator = normalized_vectors * normalized_vectors[i]

                s = numerator / denominator
                similarity_matrix[i, :] = s

        return similarity_matrix

    @staticmethod
    def sim_jaccard(df, recommender):
        """
        Calculates the jaccard similarity between two given sets(vectors) (usful for binary ratings in interaction matrix)
        :param df: sparse matrix with shape (user,item)
        :return s: similarity value between -1 and 1 (1 high correlation, 0 no correlation, -1 high negative correlation)
        """
        if recommender == 'item':
            # initalize empty similarity_matrix:
            length = df.shape[0]  # length of user vector
            item_length = df.shape[1]  # length of item vector

            similarity_matrix: ndarray = np.zeros((length, length), dtype=np.float32)  # empty similarity matrix

            for i in np.arange(length):
                # calculate item overlap between user i and every other user (list of lists)
                numerator = [
                    len([index for index, v, u in zip(np.arange(item_length), df[:, i], df[:, x]) if v > 0 and u > 0])
                    for x in np.arange(length)]
                denominator = [len(np.unique(np.array(
                    [index for index, v, u in zip(np.arange(item_length), df[:, i], df[:, x]) if v > 0 or u > 0]))) for
                               x in np.arange(length)]
                s = [numerator[i] / denominator[i] for i in np.arange(len(numerator))]
                similarity_matrix[i, :] = s

        elif recommender == 'user':
            # initalize empty similarity_matrix:
            length = df.shape[1]  # length of user vector
            item_length = df.shape[0]  # length of item vector

            similarity_matrix = np.zeros((length, length), dtype=np.float32)  # empty similarity matrix

            for i in np.arange(length):
                # calculate item overlap between user i and every other user (list of lists)
                numerator = [
                    len([index for index, v, u in zip(np.arange(item_length), df[:, i], df[:, x]) if v > 0 and u > 0])
                    for x in np.arange(length)]
                denominator = [len(np.unique(np.array(
                    [index for index, v, u in zip(np.arange(item_length), df[:, i], df[:, x]) if v > 0 or u > 0]))) for
                               x in np.arange(length)]
                s = [numerator[i] / denominator[i] for i in np.arange(len(numerator))]
                similarity_matrix[:, i] = s

        return similarity_matrix

    @staticmethod
    def sim_pearson(df, recommender):
        """
        in the making!
        """
        return warnings.warn("Similarity Method 'pearson' not yet supported. Coming soon tho!")

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
        df_train, df_test = None, None
        return df_train, df_test

    def recommend_table(self, nr_of_items, mode, method, recommender, sim='cosine'):
        path = self._da.get_nav() + 'recommendation/' + method + '_' + mode + '_' + recommender + '_recommendation.csv'
        try:
            # Read from csv
            df = pd.read_csv(path)
        except:
            matrix = self.similarity(method=method, mode=mode, sim=sim, recommender=recommender)

            # Sets diagonal to zero (if we dont want to recomend the item the user has just bought)
            np.fill_diagonal(matrix, -2)

            # gets two list of item index and item similarity rating
            nr_of_rows = matrix.shape[0]
            index = np.zeros((nr_of_rows, nr_of_items))
            ratings = np.zeros((nr_of_rows, nr_of_items))
            for row in range(nr_of_rows):
                index[row, :] = matrix[row].argsort()[-nr_of_items:][::-1].tolist()
                ratings[row, :] = matrix[row, index[row, :].astype(int)]

            tags = self.product_names(method=method)

            # Create dataframe
            df_products = pd.DataFrame(index.astype(int),
                                       columns=(['Match {}.'.format(s) for s in np.arange(1, nr_of_items + 1, 1)]))
            df_products.insert(0, "Recommendation for product:", df_products.index)
            df_similarity = pd.DataFrame(ratings, columns=(
                ['Similarity {}.'.format(s) for s in np.arange(1, nr_of_items + 1, 1)]))
            df = pd.concat([df_products, df_similarity], axis=1, sort=False)
            for i in range(len(tags)):
                df = df.replace(i, tags.iloc[i][0])

            # Write to csv
            df.to_csv(path, index=False, header=True)
            
        return df

    def single_recommend(self, product_name, nr_of_items, method, mode, recommender):
        # Read from csv
        df = self.recommend_table(nr_of_items=nr_of_items, mode=mode, method=method, recommender=recommender)

        item_id = np.where(df["Recommendation for product:"] == product_name)[0][0]

        # print results
        print("Recommendation for {}: \n".format(df.iloc[item_id][0]))
        for i in range((df.shape[1] // 2)):
            print("{}: {} with a similarity rating of {} ".format((i + 1), df.iloc[item_id][i + 1],
                                                                  round(df.iloc[item_id][df.shape[1] // 2 + i + 1], 3)))
        
        return item_id


if __name__ == '__main__':
    rs = RS()
    rs.get_interaction()
    # rs.similarity(mode='count', method='rating', sim='cosine', recommender='item')
    rs.single_recommend(product_name="#2 Coffee Filters", nr_of_items=15, method='freq', mode='rating',recommender='item')

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
