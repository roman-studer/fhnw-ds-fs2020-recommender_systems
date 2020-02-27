# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:25:15 2020

@author: Lukas
"""

class DA(object):
    
    #these two df are stored in a static way because I expect that they will be used very frequently in the Notebook chunks
    _df_origin = None
    _df_sub = None
    
    def __init__(self):
        #load data from
        DA._df_origin = df_origin
        
    def _create_df_sub(self):
        """Creates the specified sub-dataset"""
        #create df out of DA._df_origin (use self._get_df_origin()) in order to create the RS
        
    def _get_df_origin(self):
        """Lazy loader of the whole dataset"""
        if DA._df_origin == None:
            #load and store data from file
        return DA._df_origin
    
    def _get_df_sub(self):
        """Lazy loader of the sub dataset"""
        #if DA._df_sub is None, check if file exists, if yes, load and store it, otherwise create it with the according function