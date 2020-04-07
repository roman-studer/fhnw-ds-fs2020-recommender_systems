# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
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
        path = self.DA.get_nav() + method + '_' + mode + '_' + recommender + '_interaction.pkl'
        matrix = method + '_' + mode + '_' + recommender
        if os.path.exists(path):
            self._interaction[matrix] = pickle.load(open(path, "rb"))

        # create interaction_matrix
        else:
            # get data
            df_sub = self.DA.get_df_sub(method)

            # create interaction matrix
            user_c = CategoricalDtype(sorted(df_sub.user_id.unique()), ordered=True)
            product_name_c = CategoricalDtype(sorted(df_sub.product_name.unique()), ordered=True)

            row = df_sub.user_id.astype(user_c).cat.codes
            col = df_sub.product_name.astype(product_name_c).cat.codes
            if mode == 'count' or mode == 'binary':
                val = [1] * len(col)
            elif mode == 'rating':
                #creates the number of orders per unique product
                df_o = df_sub.groupby(by = ['user_id', 'product_name']).count().rename(columns = {'order_id':'o'}).reset_index()
                #creates the total number of orders from a user
                o_tot = df_sub.groupby(by = ['user_id'])['order_id'].nunique()
                #creates a new dataframe where user_id, product_id, o, o_tot
                df_rating = df_o.join(o_tot, on = 'user_id').rename(columns = {'order_id':'o_tot'})
                #generates the rating for val
                o = df_rating.loc[:, 'o'].to_numpy()
                o_tot = df_rating.loc[:, 'o_tot'].to_numpy()
                val = rating(o, o_tot).T
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
                df[df.nonzero()] = 1  # sets every value in the interaction matrix to 1 if value > 0

            self._interaction[matrix] = df

            # save interaction matrix
            pickle.dump(df, open(path, "wb"))

        return self._interaction[matrix]

    def product_names(self, method='freq'):
        """
        returns a list of products contained in the df. The index is the corresponding index number in the similarity matrix.
        """
        self.DA.get_df_sub(method)
        df = self.DA._df_sub_data[method]
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
        path_sim = self.DA.get_nav() + method + '_' + mode + '_' + recommender + '_similarity.pkl'
        if os.path.exists(path_sim):
            similarity_matrix = pickle.load(open(path_sim, 'rb'))

        # check if interaction matrix already exists:
        else:
            path = self.DA.get_nav() + method + '_' + mode + '_' + recommender + '_interaction.pkl'
            matrix = method + '_' + mode + '_' + recommender
            if os.path.exists(path):
                self._interaction[matrix] = pickle.load(open(path, "rb"))
            else:
                # load correct df here
                self.get_df_interaction(method, mode, recommender)

            similarity_matrix = self._similarity_method[sim](self._interaction[matrix], recommender)

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
        return df_train, df_test

    def fit(self, df_train):
        return None

    def recommend_table(self, nr_of_items, mode, method, recommender, sim='cosine'):
        try:
            # Read from csv
            df = pd.read_csv(method + '_' + mode + '_' + recommender + '_' + 'recommendation.csv')
        except:
            path = self.DA.get_nav() + method + '_' + mode + '_' + recommender + '_similarity.pkl'
            if os.path.exists(path):
                matrix = pickle.load(open(path, "rb"))
            else:
                self.similarity(method=method, mode=mode, sim=sim, recommender=recommender)
                print("Create similarity matrix")
                matrix = pickle.load(open(path, "rb"))

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
            df.to_csv(method + '_' + mode + '_' + recommender + '_' + 'recommendation.csv', index=False, header=True)
        return

    def single_recommend(self, product_name, nr_of_items, method, mode, recommender):
        # Read from csv
        try:
            df = pd.read_csv(method + '_' + mode + '_' + recommender + '_' + 'recommendation.csv')
        except:
            self.recommend_table(nr_of_items=nr_of_items, mode=mode, method=method, recommender=recommender)
            df = pd.read_csv(method + '_' + mode + '_' + recommender + '_' + 'recommendation.csv')

        item_id = np.where(df["Recommendation for product:"] == product_name)[0][0]

        # print results
        print("Recommendation for {}: \n".format(df.iloc[item_id][0]))
        for i in range((df.shape[1] // 2)):
            print("{}: {} with a similarity rating of {} ".format((i + 1), df.iloc[item_id][i + 1],
                                                                  round(df.iloc[item_id][df.shape[1] // 2 + i + 1], 3)))
        return

    def get_df_interaction(self, method, mode, recommender):
        """Lazy loader of the interaction matrix"""
        matrix = method + '_' + mode + '_' + recommender
        if not isinstance(self._interaction[matrix], (csc_matrix, csr_matrix)):
            path = self.DA.get_nav() + matrix + '_interaction.pkl'
            if os.path.exists(path):
                self._interaction[matrix] = pickle.load(open(path, 'rb'))
            else:
                self._interaction[matrix] = self.get_interaction(method, mode, recommender)

        return self._interaction[matrix]
    

@np.vectorize
def rating(o, o_tot, m = 10, omega = 1/3):
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


if __name__ == '__main__':
    rs = RS()
    rs.get_interaction()
    # rs.similarity(mode='count', method='rating', sim='cosine', recommender='item')
    rs.single_recommend(product_name="#2 Coffee Filters", nr_of_items=15, method='rating', mode='count',recommender='item')

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
