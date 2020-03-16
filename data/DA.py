# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas
"""
import pandas as pd
import os


class DA(object):
    
    _da = None
    
    @staticmethod
    def get_DA(nav = None):
        """
        Returns single instance of DA (DataAccess)
        
        Keyword arguments:
        nav -- path-navigation to root (default ../)
        """
        if not isinstance(DA._da, DA):
            DA._da = DA()
        if isinstance(nav, str):
            DA._da.set_nav(nav+'data/')
        elif not isinstance(DA._da.get_nav(), str):
            DA._da.set_nav('../data/')
        return DA._da
    
    def __init__(self):
        self._df_origin = None
        self._nav = None
        freq, aisle, rating = 'freq', 'aisle', 'rating'
        self._df_sub_data = {freq : None, aisle : None, rating : None}
        self._df_sub_methods = {freq : self._red_prod_freq,
                                aisle : self._red_prod_aisle,
                                rating : self._red_prod_rating}
        
    def _red_prod_freq(self, df = None, drop = 0.8):
        """Keeps `keep` of the most popular products"""
        if not isinstance(df, pd.DataFrame):
            df = self._get_df_origin()
        products = df.loc[:,'product_name'].value_counts()
        ix = int(products.keys().shape[0] * (1 - drop))
        items = products[:ix].keys()
        df_selected = df[df['product_name'].isin(items)]
        return df_selected.reset_index(drop = True)
    
    def _red_prod_aisle(self, drop = 0.8):
        """Keeps `keep` of the most popular products per aisle"""
        df = self._get_df_origin()
        df_selected = pd.DataFrame(columns = df.columns)
        df_grouped = df.groupby(['aisle_id'])
        for aisle_id, group in df_grouped:
            df_selected = df_selected.append(self._red_prod_freq(group, drop))
        return df_selected
    
    def _red_prod_rating(self):
        pass
        
    def _get_df_origin(self):
        """Lazy loader of the whole dataset"""
        if not isinstance(self._df_origin, pd.DataFrame):
            self._df_origin = pd.read_csv(self._nav+'Recommender4Retail.csv')
            self._df_origin = self._df_origin.drop(columns = [self._df_origin.columns[0]])
        return self._df_origin
    
    def get_df_sub(self, method = 'freq'):
        """Lazy loader of the sub dataset"""
        if not isinstance(self._df_sub_data[method], pd.DataFrame):
            path = self._nav+method+'.csv'
            if os.path.exists(path):
                self._df_sub_data[method] = pd.read_csv(path)
            else:
                self._df_sub_data[method] = self._df_sub_methods[method]()
                self._df_sub_data[method].to_csv(path, index = False)
        return self._df_sub_data[method]
    
    def set_nav(self, nav):
        self._nav = nav
        
    def get_nav(self):
        return self._nav
    
    
    
    
    
