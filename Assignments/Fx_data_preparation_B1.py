# -*- coding: utf-8 -*-
"""
Data prearation for 1b
@author: Abyel
"""
import numpy as np
import pandas as pd

####################### - Helper functions

"""
Splits the data into two random, equally-sized data sets. Stratified sampling
is used to make both samples contains equal probability models to get instances 
with one of the class labels e.g. "1" or "0" 
"""
def data_split(df, Class_name):
    df1 = df.copy()
    
    labels_0_indexes = np.where(df1[Class_name].values == 0)[0]
    labels_0_indexes_sets = np.random.choice(labels_0_indexes, (2,int(len(labels_0_indexes)/2)), replace=False)
    
    labels_1_indexes = np.where(df1[Class_name].values == 1)[0]
    labels_1_indexes_sets = np.random.choice(labels_1_indexes, (2,int(len(labels_1_indexes)/2)), replace=False)
    
    data_sets = zip(labels_1_indexes_sets,labels_0_indexes_sets)
    data_sets_list = list(data_sets)

    # list for both datasets
    df_list = []

    # split the dataset into 2 using indexes
    for elem in data_sets_list:
        df2 = pd.concat([df1.iloc[elem[0]], df1.iloc[elem[1]]])       
        df2_reshuffled_indexes = np.random.choice(len(df2),len(df2),replace=False)
        data_set = [df2.iloc[i] for i in df2_reshuffled_indexes]
        df3 = pd.DataFrame(data_set, columns = list(df1))
        df3.index = range(len(df3.index))
        df_list.append(df3)
        
    test = df_list[1]
    training = df_list[0]
    
    return training, test

####################### - Data preparation

dataset_name = "smiles_one_hot.csv"
Class_label_name = "active" # set class label name here
print("Data set: " + dataset_name + ", class label: " + Class_label_name)
print()

# Get the dataset, switch here with different datasets
data_set = pd.read_csv(dataset_name)
ON_data_set = data_set.copy()

# Check the amount classes between the two
class1 = sum(ON_data_set[Class_label_name].values == 0)
class2 = sum(ON_data_set[Class_label_name].values == 1)
print("Amount instances with the following classes")
print("0: " + str(class1))
print("1: " + str(class2))

# the amount features
features = len(ON_data_set.columns)
print("Amount features: " + str(features))
print()

# Split into two sets, one for cross-validation and one for testing (evaluation)
training_set, test_set = data_split(ON_data_set, Class_label_name)

## distribution
train_class0 = sum(training_set[Class_label_name].values == 0)
train_class1 = sum(training_set[Class_label_name].values == 1)
test_class0 = sum(test_set[Class_label_name].values == 0)
test_class1 = sum(test_set[Class_label_name].values == 1)

print("Class distribution")
print("training set --- 0: " + str(train_class0) + ", 1: " + str(train_class1))
print("test set --- 0: " + str(test_class0) + ", 1: " + str(test_class1))

# Save into csv
training_set.to_csv("B_training_set.csv", index=False)
test_set.to_csv("B_sampling_set.csv", index=False)








