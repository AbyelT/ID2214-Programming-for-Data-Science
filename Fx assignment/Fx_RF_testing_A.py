# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:27:31 2022
@author: Abyel
"""
import numpy as np
import pandas as pd
import past_ass_v2 as pas
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

############# Init

# Set the seed and the best hyper-parameters
#np.random.seed(100)
no_trees, criteria, no_features = 50, "gini", 2
Class_label_name = 'stroke'

# Get both training and test sets
test_set = pd.read_csv("test_set.csv")
train_set = pd.read_csv("training_set.csv")

# Create the X and Y datasets for training the models
X_train = train_set.copy()
Y_train = X_train[Class_label_name].astype('category')

############# Data preparation

X_train.drop(columns=[Class_label_name], inplace=True)
X_train, column_filter = pas.create_column_filter(X_train)
X_train, imputation = pas.create_imputation(X_train)
X_train, one_hot = pas.create_one_hot(X_train)

# Training the best performing model and baseline model on the train set
rf_hyper_parameters = RandomForestClassifier(n_estimators=no_trees, criterion=criteria, max_features=no_features)
rf_hyper_parameters.fit(X_train, Y_train)
rf_base_line = RandomForestClassifier()
rf_base_line.fit(X_train, Y_train)

# Prepare the test set, both X and Y 
results = []
X_test = test_set.copy()
Y_true = X_test[Class_label_name].astype('category')
X_test.drop(columns=[Class_label_name], inplace=True)

# Apply data preparation
X_test = pas.apply_column_filter(X_test, column_filter)
X_test = pas.apply_imputation(X_test, imputation)
X_test = pas.apply_one_hot(X_test, one_hot)

############# Testing (evaluation)
y_pred_hyper = rf_hyper_parameters.predict(X_test)
y_pred_baseline = rf_base_line.predict(X_test)

# Get the accuracy
accuracy_hyper_parameters = round(accuracy_score(Y_true, y_pred_hyper), 10)
accuracy_base_line = round(accuracy_score(Y_true, y_pred_baseline), 10)
results.append([accuracy_hyper_parameters, accuracy_base_line])

print("Evaluation:")
if accuracy_hyper_parameters > accuracy_base_line:
    print('Hyper-parameters is better')
else:
    print('baseline model is better or equal')
print('Accuracy hyper-par:', accuracy_hyper_parameters, ', trees:', no_trees, 
      ', criterion:', criteria ,', features:', no_features)
print('Accuracy baseline: ', accuracy_base_line, ', trees', rf_base_line.n_estimators, 
      ', criterion:', rf_base_line.criterion ,', features:', rf_base_line.max_features)



