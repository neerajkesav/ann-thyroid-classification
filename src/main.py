#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main.

@author: neeraj
"""
from dataset_loader import DatasetLoader
from data_preprocessor import DataPreprocessor
from model_builder import ModelBuilder

      
#Loading dataset  
data_loader = DatasetLoader('../resources/ann-train.data')
train = data_loader.load()

data_loader = DatasetLoader('../resources/ann-test.data')
test = data_loader.load()

#Preprocessing data
dp = DataPreprocessor()
train, test = dp.preprocess(train, test)

#Splitting data to predictors and target vaiables
train_X, train_y = dp.split_predictors(train)
test_X, test_y = dp.split_predictors(test)


#splitting data for validation set
X_train, X_val, y_train, y_val = dp.validation_split(train_X, train_y)

#scaling train and validation data
X_train, X_val = dp.scale_data(X_train, X_val)

#model is defined in the ModelBuilder class.
mb = ModelBuilder()
classifier = mb.get_classifier()        
       
#cross-validation on smaller set of training data
mb.validate(classifier, X_train, y_train)
##Cross Validation - Accuracy : 98.11% (1.13%)

#evaluation model using validation set
mb.evaluate(classifier, X_train, y_train, X_val, y_val)
##Accuracy is 99.073% 

#scaling train and test data
train_X, test_X = dp.scale_data(train_X, test_X)

#cross-validation on complete train data
mb.validate(classifier, train_X, train_y)
##Cross Validation - Accuracy : 98.57% (0.41%)

#Train with complete train data
classifier.fit(train_X, train_y, batch_size = 10, epochs = 100)

#predicting on test data
mb.check_prediction(classifier, test_X, test_y)
#Test Data - Accuracy is 98.279% 

#Saving model to disk
mb.save_model(classifier, '../model/final_model1')

#Save predictions
mb.save_predictions(classifier, test_X)     