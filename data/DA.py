# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas
"""
import pandas as pd


class DA(object):
    
    _da = None
    
    @staticmethod
    def get_DA():
        #singleton pattern
        if not isinstance(DA._da, DA):
            DA._da = DA()
        return DA._da
    
    def __init__(self):
        self._df_origin = None
        self._df_sub = None
        
    def _create_df_sub(self):
        """Creates the specified sub-dataset"""
        #create df out of self._df_origin (use self._get_df_origin()) in order to create the RS
        return None
        
    def _get_df_origin(self, nav = '../'):
        """Lazy loader of the whole dataset"""
        if not isinstance(self._df_origin, pd.DataFrame):
            self._df_origin = pd.read_csv(nav+'data/Recommender4Retail.csv')
        return self._df_origin
    
    def _get_df_sub(self):
        """Lazy loader of the sub dataset"""
        #if self._df_sub is None, check if file exists, if yes, load and store it, otherwise create it with the according function
        return self._df_sub
    
    def get_df_example(self):
        """Just an example function, which creates/gets the data via data-wrangling or object loading"""
        return None
