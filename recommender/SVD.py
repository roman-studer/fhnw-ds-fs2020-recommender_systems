# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:08:48 2020

@author: Lukas
"""

from recommender._Recommender_Init import _RecommenderInit
from data.DA import DA

class RSSVD(_RecommenderInit):
    
    def __init__(self):
        super().__init__()
        self._da = DA.get_DA()
        
    