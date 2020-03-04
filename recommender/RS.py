# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas, Roman
"""
from recommender._Recommender_Init import _RecommenderInit


class RS(_RecommenderInit):
    """
    Example class of a recommender. Every Recommender will have its own class
    structured.
    """

    def __init__(self, param, data):
        super().__init__()  # maybe not necessary
        self.data = data
        self._param = param  # maybe store some parameter when initializing object? idk

    def transform(self, data):
        # TO DO: correct placement of function?
        # used to transform data into a fitting format for the recommender
        pass

    def train_test(self, data):
        # TO DO: maybe move to _RecommenderInit
        # splits data into train_test if necessary
        return df_train, df_test

    def fit(self, df_train):
        return None

    def recommend(self, user):
        # user may be a list or object?

        return prediction
