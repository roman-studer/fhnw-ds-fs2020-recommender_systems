# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas
"""

class RS(object):
    
    def __init__(self, param):
        self._param = param #maybe store some parameter when initializing object? idk
        
    def fit(self, df_train):
        return None
        
    def recommend(self, user):
        #user may be a list or object?
        return prediction