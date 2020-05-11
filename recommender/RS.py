# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle, json

from numpy.core._multiarray_umath import ndarray
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, issparse, vstack
from scipy.sparse.linalg import norm
from pandas.api.types import CategoricalDtype
from recommender._Recommender_Init import _RecommenderInit
from data.DA import DA
import os
import warnings
# np.seterr(divide='ignore', invalid='ignore')



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
        path_prefix = self._da.get_nav() + 'interaction/'
        filename_prefix = method + '_' + mode + '_' + recommender
        path = path_prefix + filename_prefix + '_interaction.pkl'
        test_path = path_prefix + 'test/' + filename_prefix + '_test_interaction.pkl'
        if os.path.exists(path):
            interaction_matrix = pickle.load(open(path, "rb"))

        # create interaction_matrix
        else:
            df_sub = self._da.get_df_sub(method)

            # creates the number of orders per unique product
            df_o = df_sub.groupby(by=['user_id', 'product_name']).count().rename(
                columns={'order_id': 'o'}).reset_index()
            # creates the total number of orders from a user
            o_tot = df_sub.groupby(by=['user_id'])['order_id'].nunique()
            # creates a new dataframe where user_id, product_id, o, o_tot
            df_orders = df_o.join(o_tot, on='user_id').rename(columns={'order_id': 'o_tot'})
            # generates the rating for val
            o = df_orders.loc[:, 'o'].to_numpy()

            if mode == 'count':
                val = o
            elif mode == 'binary':
                val = [1] * len(o)
            elif mode == 'rating':
                o_tot = df_orders.loc[:, 'o_tot'].to_numpy()
                rating_fun = np.vectorize(self._rating)
                val = rating_fun(o=o, o_tot=o_tot).T
            else:
                raise AssertionError(f'Parameter mode needs to be str "cont", "binary" or "rating" not {mode}')

            user_c = CategoricalDtype(sorted(df_orders.user_id.unique()), ordered=True)
            product_name_c = CategoricalDtype(sorted(df_orders.product_name.unique()), ordered=True)

            row = df_orders.user_id.astype(user_c).cat.codes
            col = df_orders.product_name.astype(product_name_c).cat.codes

            # csr_matrix for user-user or csc_matrix for item-item
            if recommender == 'user':
                interaction_matrix = csr_matrix((val, (row, col)),
                                                shape=(user_c.categories.size, product_name_c.categories.size))
            elif recommender == 'item':
                interaction_matrix = csc_matrix((val, (row, col)),
                                                shape=(user_c.categories.size, product_name_c.categories.size))
            else:
                raise AssertionError(f'Parameter recommender needs to be str "user" or "item" not {recommender}')

            # train test split
            interaction_matrix, test_interaction_matrix = self.train_test(df=interaction_matrix, p=0.1)

            # stores interaction matrix and test_interaction_matrix
            pickle.dump(interaction_matrix, open(path, "wb"))
            pickle.dump(test_interaction_matrix, open(test_path, "wb"))  # test interaction_matrix

            # stores the according dependencies of products and users as {index : user_id/product_name}
            products = dict(enumerate(product_name_c.categories))
            products_path = path_prefix + 'products/' + filename_prefix + '_products.json'
            users_path_train = path_prefix + 'users/' + filename_prefix + '_users.json'
            users_path_test = path_prefix + 'test/' + filename_prefix + '_test_users.json'
            users_train = dict(enumerate(user_c.categories[:interaction_matrix.shape[0]]))
            users_test = dict(enumerate(user_c.categories[interaction_matrix.shape[0]:]))
            json.dump(products, open(products_path, 'w'))
            json.dump(users_train, open(users_path_train, 'w'))
            json.dump(users_test, open(users_path_test, 'w'))

        return interaction_matrix

    def get_test_interaction(self, method, mode, recommender):
        """
        Gets test interaction matrix from
        :param method: selects the method to reduce the dataframe (see product description)
        :param mode: defines how the interaction of the customer with the product should be represented (binary, count)
        :param recommender: defines if output is for a user-user matrix or a item-item matrix (csr or csc matrix)
        :return: numpy array
        """
        # check if interaction matrix already exists:
        path_prefix = self._da.get_nav() + 'interaction/test/'
        filename_prefix = method + '_' + mode + '_' + recommender
        path = path_prefix + filename_prefix + '_test_interaction.pkl'

        test_interaction_matrix = pickle.load(open(path, "rb"))

        return test_interaction_matrix

    # todo train test split 90/10
    @staticmethod
    def train_test(df, p):
        """
        Split matrix into train and testset to p/1-p proportions
        :param df: matrix
        :param p: percentage to split, e.g. 0.1 = 90/10 split
        :type p: float
        """
        train = df[:int(df.shape[0] * (1 - p)), :]
        test = df[int(df.shape[0] * (1 - p)):, :]

        return train, test

    def _rating(self, o, o_tot, m=10, omega=1 / 3):
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
            x = omega + (1 - omega) * w_prod * w_freq
        return x

    def get_interaction(self, method='freq', mode='binary', recommender='item'):
        """Lazy loader of the interaction matrix"""
        if isinstance(self._interaction_matrix, (csc_matrix, csr_matrix)) and \
                method == self._interaction_method and mode == self._interaction_mode and recommender == self._interaction_recommender:
            pass  # nothing to Do here, correct matrix is already in self._interaction_matrix
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
        :param recommender: differenctiates between ubcf or ibcf recommender
        :param df: sparse matrix with shape (user,item)
        :return s: similarity value between -1 and 1 (1 high correlation, 0 no correlation, -1 high negative correlation)
        """
        if recommender == 'item':

            # initialize empty diagonal matrix
            length = df.shape[1]
            similarity_matrix = np.zeros((length, length), dtype=np.float32)  # empty similarity matrix

            # precalculate normalized vector over whole df
            print('vektor', df[:,5145])

            normalized_vectors = norm(df, axis=0)
            print(normalized_vectors[5144])
            print(np.where(normalized_vectors==0))
            for i in np.arange(length):
                # cosine similarity calculation
                a = df[:, i]
                numerator = np.array(df.T.dot(a).todense().T)[0] # get the dotproduct for vector a and every other vector
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

    def n_nearest_items(self, nr_of_items, mode, method, recommender, sim='cosine'):
        path = self._da.get_nav() + 'similar_items/' + method + '_' + mode + '_' + recommender + '_similar_objects.csv'

        # Check if file already exists and if enought similar products are contained
        if os.path.exists(path) and (pd.read_csv(path).shape[1] // 3 >= nr_of_items):
            df = pd.read_csv(path)

        # Create file if it doesn't exists
        else:
            similarity_matrix = self.similarity(method=method, mode=mode, sim=sim, recommender=recommender)

            # Sets diagonal to -2 (if we dont want to recomend the item the user has just bought)
            np.fill_diagonal(similarity_matrix, -2)

            # gets two list of item index and item similarity rating
            nr_of_rows = similarity_matrix.shape[0]
            index = np.zeros((nr_of_rows, nr_of_items))
            similarity = np.zeros((nr_of_rows, nr_of_items))
            for row in range(nr_of_rows):
                index[row, :] = similarity_matrix[row].argsort()[-nr_of_items:][::-1].tolist()
                similarity[row, :] = similarity_matrix[row, index[row, :].astype(int)]

            tags = self.product_names(method=method)

            # Create dataframe
            df_products = pd.DataFrame(index.astype(int),
                                       columns=(['Product {}.'.format(s) for s in np.arange(1, nr_of_items + 1, 1)]))
            df_products.insert(0, "Recommendation for product:", df_products.index)
            for i in range(len(tags)):  # Replace product id with product names
                df_products = df_products.replace(i, tags.iloc[i][0])

            df_id = pd.DataFrame(index.astype(int),
                                 columns=(['id {}.'.format(s) for s in np.arange(1, nr_of_items + 1, 1)]))

            df_similarity = pd.DataFrame(similarity, columns=(
                ['Similarity {}.'.format(s) for s in np.arange(1, nr_of_items + 1, 1)]))
            df = pd.concat([df_products, df_id, df_similarity], axis=1, sort=False)
            df.insert(0, 'id', df_products.index)

            # Write to csv
            df.to_csv(path, index=False, header=True)
        return df

    def single_product(self, product_name, nr_of_items, method, mode, recommender, sim='cosine'):
        # Read from csv
        df = self.n_nearest_items(nr_of_items=nr_of_items, mode=mode, method=method, recommender=recommender, sim=sim)

        item_id = np.where(df["Recommendation for product:"] == product_name)[0][0]

        # print results
        print("Recommendation for {}: \n".format(df.iloc[item_id][0]))
        for i in range(nr_of_items):
            print("{}: {} with a similarity rating of {} ".format((i + 1), df.iloc[item_id][i + 1], round(
                df.iloc[item_id][(df.shape[1] // 3) * 2 + i + 2], 3)))

        return item_id

    def predict(self, R, S, nr_of_items, mode, method, sim='cosine'):
        """
        IBCF predictions for each given user in `R`
        
        :param R: users to predict recommendations, usually the interaction matrix (user/item), either: pd.DataFrame, np.ndarray, np.matrix or scipy.sparse matrix
        :param S: similarity matrix based on interacton matrix
        :param nr_of_items: number of similar items to consider (only necessary if mode is 'rating' or 'count')
        :param method: method chosen of how `R` was created. 'rating'/'count' will result in equation 2.8, 'binary' will result in equation 2.10
        
        :returns: -scipy.sparse coo_matrix if `method` was 'rating' or 'count'
                  - np.array if `method` was 'binary' (float32)
        """
        predictions = None
        recommender = 'item'
        # S = self.similarity(method, mode, sim, recommender)

        # convert R to sparse column matrix if not already done
        if isinstance(R, pd.DataFrame):
            R = csc_matrix(R.values)
        elif isinstance(R, np.matrix) or isinstance(R, np.ndarray):
            R = csc_matrix(R)
        elif issparse(R):
            pass

        if mode == 'rating' or mode == 'count':
            """This mode works if the rating is NOT unary AND
            when it is NOT possible for similarity scores to be negative when ratings are constrained to be nonnegative.
            
            Formula: p_{u,i} = (sum_{j∈S}(s(i,j)*r_{u,j})) / (sum_{j∈S}(abs(s(i,j)))) | S is a set of items similar to i
            
            Equation 2.8 shown in:
            Collaborative Filtering Recommender Systems 2010
            By Michael D. Ekstrand, John T. Riedl and Joseph A. Konstan"""

            batchsize = 10000  # tests have shown that this is a good batch size to avoid performance issues
            df_s = self.n_nearest_items(nr_of_items, mode, method, recommender, sim)
            df_ix = df_s.iloc[:, 2:]
            num_items = int(df_ix.shape[1] / 3)
            s_ix_np = df_ix.iloc[:, num_items:-num_items].to_numpy()
            sim_product = df_s.iloc[:, -num_items:].to_numpy()

            # create sparse similarity matrix where for each column the item_i just contains the k nearest similarities
            # rest is zero for matrix dot product
            col_ix = np.array([s_ix_np.shape[1] * [i] for i in range(s_ix_np.shape[0])]).ravel()
            row_ix = s_ix_np.astype(int).ravel()
            A = np.zeros(S.shape)
            A[row_ix, col_ix] = 1
            S = A * S  # hadamard product to just keep k similarities
            S = csr_matrix(S)

            # perform batchwise predictions
            i_prev = 0
            denominators = 1 / np.sum(np.absolute(sim_product), axis=1)

            for i in range(batchsize, R.shape[0] + batchsize, batchsize):
                # batch prep
                i = min(i, R.shape[0])
                batch = R[i_prev:i]

                # numerators
                batch_predictions = batch.dot(S)

                # denominators with hadamard product
                D = np.array([[denominator] * batch_predictions.shape[0] for denominator in denominators]).T
                batch_predictions = batch_predictions.multiply(D)

                # append batch to predictions
                if issparse(predictions):
                    predictions = vstack([predictions, batch_predictions])
                else:
                    predictions = batch_predictions

                # update slice index
                i_prev = i

        elif mode == 'binary':
            """This mode works only for unary scores.
            
            Formula: p_{u,i} = sum_{j∈I_u}(s(i,j)) | I_u is the user's purchase history
            
            Equation 2.10 shown in:
            Collaborative Filtering Recommender Systems 2010
            By Michael D. Ekstrand, John T. Riedl and Joseph A. Konstan"""

            # dot product works because summation of similarities which are in I_u is given if rating is unary
            # and non bought-items are weighted as zero
            I = coo_matrix(R).tocsr()
            predictions = np.float32(
                I.dot(S))  # np.float32 doubles execution time, but reduces memory requirements by half

        return predictions


if __name__ == '__main__':
    rs = RS()
    mode, method, sim, recommender, nr_of_items = 'count', 'freq', 'cosine', 'item', 20
    R = rs.get_interaction(method, mode, recommender)
    rs.train_test(R, 0.1)
    # R = R[:10000]
    # predictions = rs.predict(R, nr_of_items, mode, method, sim)
    # print(predictions)

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
