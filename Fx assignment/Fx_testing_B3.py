# -*- coding: utf-8 -*-
"""
Modelling and testing of the data sets
@author: Abyel
"""

import numpy as np
import pandas as pd
import past_ass_v2 as pas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report

################ HELPER FUNCTIONS

"""
Evaluates the given model with test_set. Uses f,i and o to apply 
filtering, imputation and one_hot encoding
"""
def evaluate_model(model, test_set, f, i, o, classname):
    X_test = test_set.copy()
    Y_true = X_test[classname].astype('category')
    X_test.drop(columns=[classname], inplace=True)
    
    # applying data preparation
    X_test = pas.apply_column_filter(X_test, f)
    X_test = pas.apply_imputation(X_test, i)
    X_test = pas.apply_one_hot(X_test, o)


    # get predictions and score with given test set
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # get the AUC and accuracy
    accuracy = round(accuracy_score(Y_true, y_pred), 6)
    try:
        AUC = round(roc_auc_score(Y_true, y_score[:, 1]), 6)
    except ValueError:
        print("ERROR AUC")
        AUC = 0
    
    return accuracy, AUC

################ MAIN

# Set seed and no trees, also set the name of the class label
np.random.seed(100)
no_trees = 100
classname = 'active'

# get training set and both test sets
training_set = pd.read_csv("B_training_set.csv")
majority_set = pd.read_csv("B_majority_test.csv")
undersample_set = pd.read_csv("B_undersample_test.csv")

# get the instances (X)
X_train = training_set.copy()
Y_train = X_train[classname].astype('category')

### Data preparation
X_train.drop(columns=[classname, 'index'], inplace=True, errors='ignore')
X_train, column_filter = pas.create_column_filter(X_train)
X_train, imputation = pas.create_imputation(X_train)
X_train, one_hot = pas.create_one_hot(X_train)

### Modelling
model = RandomForestClassifier(n_estimators=no_trees)
model.fit(X_train, Y_train)

# Prepare both test sets and evaluate the AUC and accuracy
results = []
major_accuracy, major_auc = evaluate_model(model, majority_set, column_filter, imputation, one_hot, classname)
under_accuracy, under_auc = evaluate_model(model, undersample_set, column_filter, imputation, one_hot, classname)

## print 
rows = [[major_accuracy, major_auc], [under_accuracy, under_auc]]
results_df = pd.DataFrame(rows, columns=['Accuracy', 'AUC'])
results_df.index = ['Imbalanced', 'class-balanced']
print()
print(results_df)
