## UCI Thyroid Classification - Python, Keras, scikit-learn, ANN

This project is created for classification problem on UCI-Thyroid-Disease dataset. It uses ANN to make predictions. Prediction classes are:
 * 1-Hyperthyroid
 * 2-Subnormal
 * 3-Normal


### Data Sets
 * Thyroid disease [data set][ds] in UCI repository.


### Frameworks/Libraries
 * Keras
 * scikit-learn
 
  
### Getting Started

These instructions will get you a brief idea on setting up the environment and running on your local machine for development and testing purposes. 

**Prerequisities**

- python3.5 or newer
- Keras
- scikit-learn
- numpy
- pandas



**Setup and running tests**

1. Run `python -V` to check the installation
   
2. Install all the required libraries.
           
3. Execute the following commands from terminal to run the tests:

      `python main.py` 


Note: Model accuracy - on validation: 98.57% (0.41%) and on test data: 98.279%. As per the dataset information any model with accuracy >92% is considered as a good one. Further improvement is definitely possible.

[ds]: <https://archive.ics.uci.edu/ml/datasets/thyroid+disease>






