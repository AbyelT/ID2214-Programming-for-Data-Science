"""
Modelling
@author: Abyel
"""

import numpy as np
import pandas as pd
import past_ass_v2 as pas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

############# Init

#input here!
np.random.seed(100) # Pick the same seed, for random generation
Class_label_name = 'stroke'
Total_features = 14

# Get training set
training_set = pd.read_csv("training_set.csv")

# Get the instances (X)
X_rf = training_set.copy()

# Get the labels (Y), drop then in the training set
Y = training_set[Class_label_name].astype('category')
X_rf.drop(columns=[Class_label_name], inplace=True)

############# Data preparation

# RF
X_rf, column_filter = pas.create_column_filter(X_rf)
X_rf, imputation = pas.create_imputation(X_rf)
X_rf, one_hot = pas.create_one_hot(X_rf)

# drop unecessary columns (for certain datasets)
# X_rf = X_rf.drop(['id'], axis=1)

############# Modelling

# Prepare cross-validation
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Prepare hyper-parameters

num_trees = [1,10,50,100,250]
criterion = ["gini", "entropy"]
max_f = range(Total_features + len(one_hot))[1: 11]

scores = []
hyper_parameters = []

# Do cross-validation for RF
for num in num_trees:
    for crit in criterion:
        for no_featues in max_f:
            model = RandomForestClassifier(n_estimators=num, criterion=crit, max_features=no_featues)
            new_sc = np.mean(cross_val_score(model, X_rf, Y, scoring="accuracy", cv=cv, n_jobs=-1))
            hyper_parameters.append({"trees": num, "critera": crit, "features": no_featues})
            scores.append(new_sc)
        
# Find the best performing model based on score (and the best configuration)
indx = np.argmax(scores)
best_parameters = hyper_parameters[indx]
    
# Create a baseline model & validate it
model = RandomForestClassifier()
base_sc = np.mean(cross_val_score(model, X_rf, Y, scoring="accuracy", cv=cv, n_jobs=-1))

# Optional data: check the amount hyper-parameter setting that gives better accuracy than the baseline
better_than_base = sum(scores > base_sc)

print("Modelling & cross-validation:")
if scores[indx] > base_sc:
    print('Hyper-parameters is better')
else:
    print('baseline model is better or equal')
print('best hyper-parameters: ', best_parameters)
print('hyper-parameters score: ', round(scores[indx], 6))
print('base model score: ', round(base_sc, 6))
print('no. hyper-parameters better than baseline model: ', better_than_base)

