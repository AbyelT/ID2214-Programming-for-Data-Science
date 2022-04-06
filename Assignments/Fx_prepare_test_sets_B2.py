# -*- coding: utf-8 -*-
"""
Preparation of the majority and equal-sized samples described in 1b
@author: Abyel
"""
import numpy as np
import pandas as pd

####################### - Helper functions

## Creates a dataset in which the majority class is equal the minority
def equal_sampling(df, classname, Half_Class_label):
    df1 = df.copy()
    labels_0_indexes = np.where(df[classname].values == 0)[0]
    labels_0_indexes_sets = np.random.choice(labels_0_indexes, Half_Class_label, replace=False)       # 211 or 528
    
    labels_1_indexes = np.where(df1[classname].values == 1)[0]
    labels_1_indexes_sets = np.random.choice(labels_1_indexes, Half_Class_label, replace=True)       # 211 or 527
    
    df2 = pd.concat([df1.iloc[labels_0_indexes_sets], df1.iloc[labels_1_indexes_sets]])       
    df2_reshuffled_indexes = np.random.choice(len(df2),len(df2),replace=False)
    data_set = [df2.iloc[i] for i in df2_reshuffled_indexes]
    df3 = pd.DataFrame(data_set, columns = list(df1))
    df3.index = range(len(df3.index))
    
    return df3

## Creates a dataset in which the majority class is represented as 4/5 in the dataset
def adjust_sampling(df, classname, Half_Class_label, four_of_five):
    df1 = df.copy()
    labels_0_indexes = np.where(df1[classname].values == 0)[0]
    labels_0_indexes_sets = np.random.choice(labels_0_indexes, four_of_five, replace=False)     # 4/5 of the size
    
    labels_1_indexes = np.where(df1[classname].values == 1)[0]
    labels_1_indexes_sets = np.random.choice(labels_1_indexes, Half_Class_label, replace=True)  # 1/5 of the size

    df2 = pd.concat([df1.iloc[labels_0_indexes_sets], df1.iloc[labels_1_indexes_sets]])       
    df2_reshuffled_indexes = np.random.choice(len(df2),len(df2),replace=False)
    data_set = [df2.iloc[i] for i in df2_reshuffled_indexes]
    df3 = pd.DataFrame(data_set, columns = list(df1))
    df3.index = range(len(df3.index))
    
    return df3

####################### - Data preparation
Class_name = "active"  # set class name here
Half_Class_label = 211 # hardcoded, represents half the amount of minority class label 
                        # e.g. 211 in smiles.one_hot, 1000 in diabetes_binary dataset
four_of_five = 844     # hardcoded, represents 4/5 of the majority class label 
                        # e.g. 844 in smiles.one_hot, 4000 in diabetes_binary dataset

# Get the test dataset, copy for majority and one for equal_sampling
data_set = pd.read_csv("B_sampling_set.csv")
major_set = data_set.copy()
equal_set = data_set.copy()

#Split into two sets, one for training and one for testing
major_set = adjust_sampling(major_set, Class_name, Half_Class_label, four_of_five)   # for 5:1 majority set
equal_set = equal_sampling(equal_set, Class_name, Half_Class_label)                  # for equal sets

#save into csv
major_set.to_csv("B_majority_test.csv", index=False)
equal_set.to_csv("B_undersample_test.csv", index=False)
