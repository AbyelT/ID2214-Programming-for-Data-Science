"""
Data preparation for task 1a
@author: Abyel
"""
import numpy as np
import pandas as pd

####################### - Helper functions

"""
Splits the data into two random, equally-sized data sets. Stratified sampling
is used to make both samples contains equal probability models to get instances 
with one of the class labels e.g. "1" or "0" 
DEPRECATED
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
    
    # determine through coin flip which set that becomes the test set and training set
    flip = np.random.choice(len(df_list), 1, replace=False)
    test = df_list[flip[0]]
    df_list.pop(flip[0])
    training = df_list[0]
    
    return training, test

"""
Splits the data into two random, equally-sized data sets. Compared to the
above, this function uses random sampling of the data set
"""
def random_split(df, Class_name):
    df1 = df.copy()
    df1_reshuffled = np.random.choice(df1.index, len(df1.index), replace=False)
    
    # create a list of two samples, both are used to create train and test sets
    two_random_indexSets = np.random.choice(df1_reshuffled, (2,int(len(df1.index)/2)), replace=False)

    # list for both datasets
    df_list = []
    
    for elem in two_random_indexSets:
        df2 = df1.iloc[elem]      
        df2_reshuffled_indexes = np.random.choice(len(df2),len(df2),replace=False)
        data_set = [df2.iloc[i] for i in df2_reshuffled_indexes]
        df3 = pd.DataFrame(data_set, columns = list(df1))
        df3.index = range(len(df3.index))
        df_list.append(df3)

    # determine through coin flip which set that becomes the test set and training set
    flip = np.random.choice(len(df_list), 1, replace=False)
    test = df_list[flip[0]]
    df_list.pop(flip[0])
    training = df_list[0]
    
    return training, test

####################### - Data preparation

#input here!
dataset_name = "healthcare-dataset-stroke-data.csv"
Class_label_name = "stroke" # set class label name here
print("Data set: " + dataset_name + ", class label: " + Class_label_name)
print()

# Get the dataset
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
training_set, test_set = random_split(ON_data_set, Class_label_name)

## distribution
train_class0 = sum(training_set[Class_label_name].values == 0)
train_class1 = sum(training_set[Class_label_name].values == 1)
test_class0 = sum(test_set[Class_label_name].values == 0)
test_class1 = sum(test_set[Class_label_name].values == 1)

print("Class distribution")
print("training set --- 0: " + str(train_class0) + ", 1: " + str(train_class1))
print("test set --- 0: " + str(test_class0) + ", 1: " + str(test_class1))

# Save into csv files
training_set.to_csv("training_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)

