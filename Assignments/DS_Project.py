import numpy as np
import pandas as pd
# from numpy.linalg import norm
from math import ceil
import time

"""learning models"""
# from kNN import kNN
# from naiveBayes import NaiveBayes
# from Out_of_bag.py import RandomForest

"""related to SMILE"""
# from rdkit import Chem
# import rdkit.Chem.rdMolDescriptors as d
# import rdkit.Chem.Fragments as f
# import rdkit.Chem.Lipinski as l

""" Code and classes for the ID2214 Data Science project,
consists of data preparation techniques (functions), learning algorithms (as classes)
with parameter setting and a function that generates models out of differnt combinations
given the input e.g. binning and inputation using kNN.

The file is organised into sections

1. Data preparation functions
2. learning algorithms (ALREADY IMPORTED)
3. function for generating models
4. main code with input
"""

"""Data preparation functions"""

def create_column_filter(df):
    new_df = df.copy()
    new_df = new_df.filter([e for e in new_df.columns if new_df[e].nunique() > 1 or e in ['CLASS', 'ID']], axis=1)
    return new_df, list(new_df.columns)

def apply_column_filter(df, column_filter):
    new_df = df.copy()
    new_df = new_df.filter(items=column_filter, axis=1)
    return new_df

def create_normalization(df, normalizationtype="minmax"):
    new_df = df.copy()
    normalization = {}
    for e in new_df.columns:
        if e not in ['CLASS', 'ID'] and new_df[e].dtypes in ["float64", "int64"]:
            if normalizationtype == "minmax":
                min = new_df[e].min()
                max = new_df[e].max()
                new_df[e] = [(x-min)/(max-min) for x in new_df[e]]
                normalization[e] = ("minmax", min, max)
            elif normalizationtype == "zscore":    
                mean = new_df[e].mean()
                std = new_df[e].std()
                new_df[e] =  [((x-mean)/std) for x in new_df[e]]
                normalization[e] = ("zscore", mean, std)
    return new_df, normalization

def apply_normalization(df, normalization):
    new_df = df.copy()
    for e in new_df.columns:
        if e in normalization:
            a = normalization.get(e)[1]
            b = normalization.get(e)[2]
            new_df[e] = [(x-a)/(b-a) for x in new_df[e]] # works with minmax or zscore
    return new_df

def create_imputation(df):
    new_df = df.copy()
    imputation = {}
    for e in new_df.columns:
        if e not in ['CLASS', 'ID']:
            # numerical values
            if new_df[e].dtypes in ["float64", "int64"]:
                if new_df[e].nunique() < 1:
                    new_df[e].fillna(0, inplace=True)
                new_df[e].fillna(new_df[e].mean(), inplace=True)
                imputation[e] = new_df[e].mean()
            # categorical or object values
            else:
                # if it is an object
                if new_df[e].dtypes == "object":
                    new_df[e] = new_df[e].astype('category')
                if new_df[e].nunique() < 1:
                    new_df[e].fillna(new_df[e][0], inplace=True)
                new_df[e].fillna(new_df[e].mode()[0], inplace=True)
                imputation[e] = new_df[e].mode()[0]
    return new_df, imputation

def apply_imputation(df, imputation):
    new_df = df.copy()
    new_df.fillna(imputation, inplace=True)
    return new_df

def create_bins(df, nobins=10, bintype="equal-width"): 
    new_df = df.copy()
    binning = {}
    for e in new_df.columns:
        if e not in ['CLASS', 'ID'] and new_df[e].dtypes in ["float64", "int64"]:
            if bintype == "equal-width":
                new_df[e], binning[e] = pd.cut(new_df[e], nobins, labels=False, retbins=True)                           #3
                new_df[e] = new_df[e].astype('category')                                                                #4
                new_df[e] = new_df[e].cat.set_categories(np.arange(nobins))                                             #5 (redundant)
            elif bintype == "equal-size":
                new_df[e], binning[e] = pd.qcut(new_df[e], nobins, labels=False, retbins=True, duplicates="drop")       #3
                new_df[e] = new_df[e].astype('category')                                                                #4
                new_df[e] = new_df[e].cat.set_categories(np.arange(nobins))                                             #5 (redundant)
            binning[e][0] = -np.inf
            binning[e][-1] = np.inf
    return new_df, binning

def apply_bins(df, binning):
    new_df = df.copy()
    for e in new_df.columns:
        if e in binning:
            new_df[e] = pd.cut(new_df[e], binning[e], labels=False, retbins=False)        #2
            new_df[e] = new_df[e].astype('category')                                      #3
            new_df[e] = new_df[e].cat.set_categories(np.arange(binning[e].size))          #4 
    return new_df

def create_one_hot(df):
    new_df = df.copy()
    handle = new_df.filter([e for e in new_df.columns if e not in ['CLASS', 'ID']], axis=1)
    one_hot = {}
    for e in handle.columns:
        if new_df[e].dtypes.name == 'category':
            features = np.sort(handle[e].unique())  # 3
            for i in features:
                new_df[e + "-" + str(i)] = [1.0 if x == i else 0.0 for x in handle[e]]
                new_df[e + "-" + str(i)].astype('float')  # 4
            one_hot[e] = features
            new_df.drop(e, axis=1, inplace=True)  # 5
    return new_df, one_hot

def apply_one_hot(df, one_hot):
    new_df = df.copy()
    for e in new_df.columns:
        if e in one_hot and e not in ['CLASS', 'ID']:
            for i in one_hot[e]:
                new_df[e + "-" + str(i)] = [1.0 if x == i else 0.0 for x in new_df[e]]
                new_df[e + "-" + str(i)].astype('float')  # 4
            new_df.drop(e, axis=1, inplace=True)  # 5
    return new_df

def accuracy(df, correctlabels):
    highest_probability = df.idxmax(axis=1)
    correct_occurances = 0
    for correct_label, predicted_label in zip(correctlabels, highest_probability):
        if correct_label == predicted_label:
            correct_occurances += 1

    return correct_occurances/df.index.size

def brier_score(df, correctlabels):
    squared_sum = 0
    row = 0
    for label in correctlabels:
        i = np.where(df.columns == label)[0]
        for col in df.columns:
            squared_sum += (1 - df.loc[row, label]
                            )**2 if label == col else df.loc[row, col]**2
        row += 1
    return squared_sum/df.index.size

def auc(df, correctlabels):
    auc = 0
    for col in df.columns:
        df2 = pd.concat(
            [df[col], pd.Series(correctlabels.astype('category'), name='correct')], axis=1)
        # get dummies for correct labels and sort descending
        df2 = pd.get_dummies(df2.sort_values(col, ascending=False))
        # move col to first for easier total tp and fp calculation
        tmp = df2.pop('correct_'+str(col))
        # get the col frequency for calculating weighted AUCs
        col_frequency = tmp.sum()/tmp.index.size
        df2.insert(1, tmp.name, tmp)
        scores = {}
        # populate the scores dictionary for column i.e. key=score, value=[tp_sum, fp_sum]
        for row in df.index:
            key = df2.iloc[row, 0]
            current = np.zeros(2, dtype=np.uint) if scores.get(
                key) is None else scores[key]
            to_add = np.array([1, 0]) if df2.iloc[row,
                                                  1] == 1 else np.array([0, 1])
            scores[key] = current+to_add

        # calculate auc based on scores
        cov_tp = 0
        column_auc = 0
        tot_tp = 0
        tot_fp = 0
        # calculate total tp and fp
        for value in scores.values():
            tot_tp += int(value[0])
            tot_fp += int(value[1])

        # same algorithm as in the lecture (bad naming though)
        for i in scores.values():
            if i[1] == 0:
                cov_tp += i[0]
            elif i[0] == 0:
                column_auc += (cov_tp/tot_tp)*(i[1]/tot_fp)
            else:
                column_auc += (cov_tp/tot_tp) * \
                    (i[1]/tot_fp)+(i[0]/tot_tp)*(i[1]/tot_fp)/2
                cov_tp += i[0]
        auc += col_frequency*column_auc
    return auc

"""Function for generating models"""

"""Generate learning model with training model, model type and optional parameters
    optional parameters: 
        norm_type: minmax|zscore
        nobins: 10
        bintype: equal-width|equal-size
        no_trees: 100
"""
def generate_model(df, modeltype="RandomForest", cross_validation=10, options={"no_trees": 100}):
    new_df = df.copy()
    equal_interval = ceil(len(new_df.index)/cross_validation) #do floor

    
    folds = [pd.DataFrame(new_df.iloc[i*equal_interval:(i+1)*equal_interval, :]) for i in range(cross_validation)]
    for k in range(cross_validation):  
        
        train_proper_model()
    
    
    #folds = [ for i in ]
    
    #split dataset into k folds OK 
    #do for loop for each fold
        #divide the k-1 training set into proper training set and validation set
        #use data prep techniques to prepare data
        #generate differnt models from data prep + learning algorithms + hyper-parameters with proper tarining set
        #test each model with the validation set
        #if good enough, measure the model wth the test set instead
        #save the performance from testing the model on an array
    #add the prediction of all models and take the average
    
    #do switch case with modeltype
    #for each case
        #1. prepare data
        #2. divide into proper training set and validation set
        #3. use cross validation algorithm to find the best validated model
    
def train_proper_model(k_folds)    
    print("the k-1 folds!")
    print(k_folds)
    
#rng = np.random.default_rng()
    #random_indexes = rng.choice(new_df.index, len(new_df.index), replace=True)    # 2. create list of random indexes for random sampling    
    
def test_model():
    print("testing!")
    #4.	Measure performance of model trained using this configuration and with 
    # the full training set, on the test set. Take the average of the performance 
    # through all k folds 
    #5.	Find the best configuration by cross-validation, generate a new model 
    # from the full data set using the same configuration 
    # (give stronger model at the cost of biased performance estimation)



feature_set1 = pd.read_csv("feature_set1.csv")
feature_set2 = pd.read_csv("feature_set2.csv")

t0 = time.perf_counter()
generate_model(feature_set1, "RandomForest", cross_validation=2)

print("Training time: {:.2f} s.".format(time.perf_counter()-t0))










## testing the libraries
# m = Chem.MolFromSmiles('Cc1ccccc1')
# print(m)
# print(m.GetNumAtoms())
# print(d.CalcExactMolWt(m))
# print(f.fr_Al_COO(m))
# print(l.HeavyAtomCount(m))

