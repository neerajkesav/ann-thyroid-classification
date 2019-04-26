#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DatasetLoader.
@author: neeraj kesavan
"""
import pandas

class DatasetLoader:
    """ Class DatasetLoader. To load dataset. DatasetLoader have the
    following properties:
    
    Attributes:
        path: path to the dataset.
    
    Methods:
        __init__(): Constructor. initialize variable path.
        load(): load datset to data from specified path and return data.
        print_shape(): print the shape of data.
    
    """
    
    path = ""
       
    def __init__(self, path):
        """Takes arguments 'path'  and initializes class variable.        
        """
        self.path = path
        
    def load(self):
        """Takes no arguments, load dataset to 'data' from path
        and returns data.
        
        data: loaded with dataset.       
        """
        data = pandas.read_csv(self.path, header=None, sep=' ')
        return data
        
    def print_shape(self, data):
        """Takes argument 'data', prints its shape.
        data: contains dataset.         
        """
        print(data.shape)
