# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas
"""
import pandas as pd
import os
from pathlib import Path


class DA(object):
    _da = None

    @staticmethod
    def get_DA(nav=None):
        """
        Returns single instance of DA (DataAccess)

        Keyword arguments:
        nav -- path-navigation to root (default ../)
        """
        if not isinstance(DA._da, DA):
            DA._da = DA()
        if isinstance(nav, str):
            DA._da.set_nav(nav + 'data/')
        elif not isinstance(DA._da.get_nav(), str):
            DA._da.set_nav('../data/')
        return DA._da

    def __init__(self):
        self._df_origin = None
        self._nav = None
        freq, aisle, rating = 'freq', 'aisle', 'rating'
        self._df_sub_data = {freq: None, aisle: None, rating: None}
        self._df_sub_methods = {freq: self._red_prod_freq,
                                aisle: self._red_prod_aisle,
                                rating: self._red_prod_rating}

    def _red_prod_freq(self, df=None, drop=0.8):
        """Drops `drop` of the most unpopular products"""
        if not isinstance(df, pd.DataFrame):
            df = self._get_df_origin()
        products = df.loc[:, 'product_name'].value_counts()
        ix = int(products.keys().shape[0] * (1 - drop))
        items = products[:ix].keys()
        df_selected = df[df['product_name'].isin(items)]
        df_selected = self.drop_user(df_selected)
        return df_selected.reset_index(drop=True)

    def _red_prod_aisle(self, drop=0.8):
        """Drops `drop` of the most unpopular products per aisle"""
        df = self._get_df_origin()
        df_selected = pd.DataFrame(columns=df.columns)
        df_grouped = df.groupby(['aisle_id'])
        for aisle_id, group in df_grouped:
            df_selected = df_selected.append(self._red_prod_freq(group, drop))
        df_selected = self.drop_user(df_selected)

        return df_selected

    def _red_prod_rating(self, drop=0.8):
        """Drops `drop` of the most unpopular products evaluated per rating"""
        df = self._get_df_origin()
        order_id, product_name, freq, support, user_id = 'order_id', 'product_name', 'freq', 'support', 'user_id'
        count_customers, customer_ratio, reordered, rating = 'count_customers', 'customer_ratio', 'reordered', 'rating'

        # creates a product df inc. support
        trans_count = df.loc[:, order_id].unique().shape[0]
        df_products = pd.DataFrame(df.loc[:, product_name].value_counts()).reset_index()
        df_products = df_products.rename(columns={df_products.columns[0]: product_name, df_products.columns[1]: freq})
        df_products[support] = df_products.loc[:, freq] * (1 / trans_count)

        # creates customer ratio
        customer_number = df.loc[:, user_id].unique().shape[0]
        df_product_customers = df.loc[:, [user_id, product_name]].groupby(product_name)
        df_product_customers = pd.DataFrame(df_product_customers.user_id.nunique())
        df_product_customers = df_product_customers.reset_index().rename(columns={user_id: count_customers})
        df_product_customers[customer_ratio] = df_product_customers.loc[:, count_customers] * (1 / customer_number)

        # creates reordered transactions and merges all calculated ratios to the product df
        df_product_reordered = df.loc[:, [product_name, reordered]].groupby(product_name).sum().reset_index()
        df_products = pd.merge(df_products, df_product_customers, on=product_name)
        df_products = pd.merge(df_products, df_product_reordered, on=product_name)

        # normalizes all ratios between 0 and 1 to give each ratio the same weight and sums them up
        df_products.loc[:, [support, customer_ratio, reordered]] -= df_products.loc[:,
                                                                    [support, customer_ratio, reordered]].min()
        df_products.loc[:, [support, customer_ratio, reordered]] /= df_products.loc[:,
                                                                    [support, customer_ratio, reordered]].max()
        df_products[rating] = df_products.loc[:, [support, customer_ratio, reordered]].sum(axis=1)

        # drops the specified amount of products out of the original df by the rating
        df_products = df_products.sort_values(by=[rating], ascending=False)
        products = df_products.loc[:, product_name]
        ix = int(products.keys().shape[0] * (1 - drop))
        items = products[:ix].values
        df_selected = df[df[product_name].isin(items)]
        df_selected = self.drop_user(df_selected)

        return df_selected.reset_index(drop=True)

    def _red_prod_rating2(self, drop=0.8):
        """Keeps the products with the highest rating"""

        # create df with columns: count of orders for product, customers per product, reorders per product:
        if Path('Recommender4Retail.csv').is_file():
            chunks = pd.read_csv("Recommender4Retail.csv", chunksize=10_000)
        else:
            raise Exception('No file named "Recommender4Retail.csv" found in directory.')

        subsets = [chunk.groupby('product_id').agg({'product_id': 'count',
                                                    'user_id': 'nunique',
                                                    'reordered': 'sum'}) for chunk in chunks]
        df = pd.concat(subsets).groupby(level=0).sum()
        df.reset_index(inplace=True, drop=True)
        df.rename(columns={"product_id": "n_orders", "user_id": "n_users", 'reordered': "n_reorders"}, inplace=True)

        # create rating:
        n_orders = sum(df['n_orders'])  # total number of ordered products
        n_customers = sum(df['n_users'])  # total number of customers
        n_reorders = sum(df['reordered'])  # total number of reorders
        df['rating'] = df['n_orders'] / n_orders + df['n_users'] / n_customers + df['n_reorders'] / n_reorders

        # normalize rating
        df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())

        # drop "bad" products:
        df.sort_values('rating', ascending=False, inplace=True)

        # calculate condition to drop
        rows_to_drop = int(len(df) * drop)
        df = df.drop(df.tail(rows_to_drop).index)

        # drop transactions
        good_products = df.index.tolist()

        chunks = pd.read_csv('Recommender4Retail.csv', chunksize=10_000)
        for chunk in chunks:
            chunk = chunk[chunk['product_id'].isin(good_products)]

            if Path('rating.csv').is_file():
                chunk.to_csv('rating.csv', mode='a', header=False)
            else:
                chunk.to_csv('rating.csv', mode='a', header=True)
        df_selected = pd.read_csv('rating.csv')
        df_selected = self.drop_user(df_selected)

        return df_selected

    def _get_df_origin(self):
        """Lazy loader of the whole dataset"""
        if not isinstance(self._df_origin, pd.DataFrame):
            self._df_origin = pd.read_csv(self._nav + 'Recommender4Retail.csv', usecols=['user_id',
                                                                                         'order_id',
                                                                                         'product_id',
                                                                                         'product_name',
                                                                                         'reordered',
                                                                                         'aisle_id'])
            # self._df_origin = self._df_origin.drop(columns=[self._df_origin.columns[0]]) # not necessary after only loading 5 special columns
        return self._df_origin

    def get_df_sub(self, method='freq'):
        """Lazy loader of the sub dataset"""
        if not isinstance(self._df_sub_data[method], pd.DataFrame):
            path = self.get_nav() + 'sub/' + method + '.csv'
            if os.path.exists(path):
                self._df_sub_data[method] = pd.read_csv(path)
            else:
                df_sub = self._df_sub_methods[method]()
                self._df_sub_data[method] = df_sub.loc[:, ['user_id', 'order_id', 'product_name']]
                self._df_sub_data[method].to_csv(path, index=False)
        return self._df_sub_data[method]

    def set_nav(self, nav):
        self._nav = nav

    def get_nav(self):
        return self._nav

    @staticmethod
    def drop_user(df, n_products=50):
        """
        Drops every user in a DataFrame that has n_orders or less in total

        :param df: sorted and reduced DataFrame
        :param n_products: Number of products of a customer, below that they will be dropped

        :return: pd.DataFrame where the customers with too less purchases are dropped
        """
        # aggregate by number of different purchased products
        df_agg = df.groupby('user_id').agg(num_prod=pd.NamedAgg(column='product_name', aggfunc='nunique')).reset_index()

        # drop every customer with number of orders <= n_orders
        users = df_agg.loc[df_agg.num_prod >= n_products].user_id.to_list()
        # reduce DataFrame
        df = df[df.user_id.isin(users)]

        return df


if __name__ == '__main__':
    A = DA.get_DA()
    A.get_df_sub(method='count')
