#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelBuilder.

@author: neeraj kesavan
"""
import pickle
import numpy
import pandas

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

class ModelBuilder:
    """Class ModelBuilder. Creates, validate, evaluate and saves the model.
    
    Methods:
        classifier_model(): defines the neural network model to be used in the scikit-learn wrapper.
        get_classifier(): gets the scikit-learn wrapped keras classifier model.
        finalize_and_save(): fits the final model and save to disk.
        save_model(): saves the model to disk.
        load_model(): loads and returns model from disk.
        check_prediction(): checks prediction accuracy.
        save_predictions(): save predictions to disk.        
    
    """
    
    def classifier_model(self):
        """Takes no arguments.
        Defines the neural network model.
        
        Returns the 'model'
        """
        #Sequential model
        model = Sequential()
        
        #Input layer and first hidden layer.
        model.add(Dense(48, kernel_initializer = 'uniform', input_dim=21, activation='relu'))
        
        #25% of neurons are droppedout to avoid over learning/fitting.
        model.add(Dropout(0.25))
        
        #2nd hidden layer
        model.add(Dense(48, kernel_initializer = 'uniform', activation='relu'))
        
        #25% of neurons are droppedout to avoid over learning/fitting.
        model.add(Dropout(0.25))
        
        #3rd hidden layer
        model.add(Dense(48, kernel_initializer = 'uniform', activation='relu'))
        
        #25% of neurons are droppedout to avoid over learning/fitting.
        model.add(Dropout(0.25))
        
        #Output layer
        model.add(Dense(3, kernel_initializer = 'uniform', activation='softmax'))
        
    	#Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    
    def get_classifier(self):
        """Takes no argument.
        Creates classifier model usig scikit-learn wrapper in Keras.
        
        Returns 'classifier'.
        """
        #Creates model
        classifier = KerasClassifier(build_fn = self.classifier_model, batch_size = 10, epochs = 100)
        
        return classifier
    
    def finalize_and_save(self, model, train_X, train_y, filename='../model/final_model'):
        """Takes arguments 'model', 'filename', 'train_X', 'train_y'.
        model: finalized model.
        filename: path+filename to which model to be saved.
        train_X: input part of train_set.
        train_y: output part of train_set.
        
        Saves the model to disk.
        """
        #Fits the model to train data
        model.fit(train_X, train_y)
        
        #Saves the model to disk
        self.save_model(model, filename)
        
    def save_model(self, model, filename='../model/saved_model'):
        """Takes arguments 'model', 'filename'.
        model: finalized model.
        filename: path+filename to which model to be saved.
        
        Saves the model to disk.
        """
        #Save the model to disk
        pickle.dump(model, open(filename, 'wb' ))
        print("\nModel is saved..\n")
    
    def load_model(self, model_filename):
        """Takes argument 'model_filename'.
        model_filename: path+filename of model to be loaded.
        
        Returns the loaded 'model'
        """
        #Load the model from disk
        loaded_model = pickle.load(open(model_filename, 'rb' ))
        
        return loaded_model
    
    def validate(self, model, train_X, train_y):
        """Takes arguments: 'model', 'train_X', 'train_y'.
        model: model to be validated.
        train_X: input part of dataset.
        train_y: output part of dataset.
        
        Perfoms cross-validation on the specified model and prints accuracy.
        """
        results = cross_val_score(estimator = model, X = train_X, y = train_y, cv = 10, n_jobs = 3)
        
        print("\nCross Validation - Accuracy : %.2f%% (%.2f%%)\n" % (results.mean()*100.0, results.std()*100.0))
        
    def evaluate(self, model, train_X, train_y, test_X, test_y):
        """Takes arguments: 'model', 'train_X', 'train_y', 'test_X', 'test_y'.
        model: model to be evaluated.
        train_X: input part of train data.
        train_y: output part of train data.
        test_X: input part of test data - validation.
        test_y: output part of test data - validation.
        
        Perfoms evaluation on the specified model and prints accuracy.
        """
        #Fits the model to traindata
        model.fit(train_X, train_y, batch_size = 10, epochs = 100) 
        
        #prediction
        y_test_pred = model.predict(test_X)
        
        #Confustion matrix from predictions
        cm = confusion_matrix(test_y, y_test_pred)
        
        print("\nModel Evaluation - Accuracy is %.3f%% \n" % ((cm[0][0]+cm[1][1]+cm[2][2])*100/test_y.size))
        
    def check_prediction(self, model, test_X, test_y):
        """Takes arguments: 'model', 'test_X', 'test_y'.
        model: model to be evaluated.
        test_X: input part of test data
        test_y: output part of test data

        Perfoms predicton and prints accuracy.
        """
        #Prediction
        y_test_pred = model.predict(test_X)
        
        #Confustion matrix from predictions
        cm = confusion_matrix(test_y, y_test_pred)
        
        #Map predictions to class name
        y_test_pred = self.map_pred_class(y_test_pred)
        
        print("\n............Predictions............\n")
        print(y_test_pred.reshape(-1,1))
        print("\nTest Data - Accuracy is %.3f%% \n" % ((cm[0][0]+cm[1][1]+cm[2][2])*100/test_y.size))
    
    def save_predictions(self, model, test_X):
        """Takes arguments: 'model', 'test_X'.
        model: model to be evaluated.
        test_X: test data input to make prediction

        Perfoms prediction and saves to disk.
        """
        #Prediction
        predictions = model.predict(test_X)

        #Map predictions to class name
        predictions = self.map_pred_class(predictions)
        
        #Saves to disk
        pandas.DataFrame(predictions).to_csv('../prediction/predictions.csv', index=False)
        print("\nPredictions are saved..\n")
    
    def map_pred_class(self, preditions):
        """Takes argument 'predictions'.
        predictions: contains precited integer class values to be mapped to actual classes.
        
        Returns predicted class names.
        """
        
        pred_map = ['Normal'  if(x==3) else 'Subnormal' if (x==2) else 'HyperThyroid'  for x in preditions]
        
        return numpy.array(pred_map)