#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataPreprocessor.
@author: neeraj
"""
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Class DataPreprocessor. To perform preprocessing on dataset.
     
    Methods:
        preprocess(): to get the clean data
        split_predictors(): splits dataset to X and y variables.
        scale_data(): Scales the train and test data
        validation_split(): splits the train data to get train and validation data
    """
    
    def preprocess(self, train, test):
        """Takes arguments 'train', 'test'.
        train: training data
        test: testing data
        
        Returns clean 'train' and 'test' data.
        """
        #This dataset from UCI is almost a clean one. So nothing much to do.
        column_list = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'Class']
        train = pd.DataFrame(train.iloc[:,0:22].values, columns=column_list)
        test = pd.DataFrame(test.iloc[:,0:22].values, columns=column_list)
        
        return train, test
    
    def split_predictors(self, data):
        """Takes argument 'data'
        data: data to split on predictors nd target variable.
        
        Returns the X and y variable wise parts of data
        """
        #Predictor data
        data_X = data.drop(['Class'], axis=1)
        
        #Target data
        data_y = data['Class']
        
        return data_X, data_y
    
    def scale_data(self, train_X, test_X):
        """Takes argument 'train_X', 'test_X'.
        train_X: predictor data for training
        test_X: predictor data for testing
        
        Returns the scaled train and test data.
        """
        sc = StandardScaler()
        #scaling train data
        train_X = sc.fit_transform(train_X)
        
        #scaling test data in the same sacle of train data.
        test_X = sc.transform(test_X)
        
        return train_X, test_X
    
    def validation_split(self, train_X, train_y, test_size = 0.2, random_state = 1):
        """Takes arguments 'train_X', 'train_y', 'test_size', 'random_state'.
        train_X: predictor data of train dataset
        train_y: target data of train dataset
        test_size: fraction of test set to be splitted from train data. Default=0.2
        random_state: Default= 1    
        
        Splits the train data to get validation dataset. 
        
        Returns train and validation data as X and y parts.
        """
        #Spliting data
        X_train, X_validation, y_train, y_validation = train_test_split(train_X, train_y, test_size = 0.2, random_state = 1)        
        
        return X_train, X_validation, y_train, y_validation
        
