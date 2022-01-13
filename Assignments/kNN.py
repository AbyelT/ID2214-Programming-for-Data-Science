import numpy as np
import pandas as pd
import time
from numpy.linalg import norm

###################FUNCTIONS FROM ASSIGN 1##########################

def create_column_filter(df):
    new_df = df.copy()
    new_df = new_df.filter([e for e in new_df.columns if new_df[e].nunique() > 1 or e in ['CLASS', 'ID']], axis=1)
    return new_df, list(new_df.columns)

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

def create_imputation(df):
    new_df = df.copy()
    imputation = {}
    for e in new_df.columns:
        # numerical values
        if e not in ['CLASS', 'ID'] and new_df[e].dtypes in ["float64", "int64"]:
            if new_df[e].nunique() < 1:
                new_df[e].fillna(0, inplace=True)
            new_df[e].fillna(new_df[e].mean(), inplace=True)
            imputation[e] = new_df[e].mean()
        #categorical or object values
        else:
            #if it is an object
            if new_df[e].dtypes == "object":
                new_df[e].astype('category')
            if new_df[e].nunique() < 1:
                new_df[e].fillna(new_df[e][0], inplace=True)
            new_df[e].fillna(new_df[e].mode()[0], inplace=True)
            imputation[e] = new_df[e].mode()[0]
    return new_df, imputation

def create_one_hot(df):
    new_df = df.copy()
    handle = new_df.filter([e for e in new_df.columns if new_df[e].dtype.name in ['object', 'category'] or e not in ['CLASS', 'ID']], axis=1)
    one_hot = {}
    for e in handle.columns:
        if handle[e].dtypes.name == "category":
            features = np.sort(handle[e].unique())                                      #3
            for i in features:  
                new_df[e + "-" + str(i)] = [1.0 if x == i else 0.0 for x in handle[e]]
                new_df[e + "-" + str(i)].astype('float')                                     #4
            one_hot[e] = features
            new_df.drop(e, axis=1, inplace=True)                                        #5
    return new_df, one_hot

def apply_column_filter(df, column_filter):
    new_df = df.copy()
    new_df = new_df.filter(items=column_filter, axis=1)  
    return new_df

def apply_normalization(df, normalization):
    new_df = df.copy()
    for e in new_df.columns:
        if e in normalization:
            a = normalization.get(e)[1]
            b = normalization.get(e)[2]
            new_df[e] = [(x-a)/(b-a) for x in new_df[e]] # works with minmax or zscore
    return new_df

def apply_imputation(df, imputation):
    new_df = df.copy()
    new_df.fillna(imputation, inplace=True)
    return new_df

def apply_one_hot(df, one_hot):
    new_df = df.copy()
    for e in new_df.columns:
        if e in one_hot and e not in ['CLASS', 'ID']:
            for i in one_hot[e]: 
                new_df[e + "-" + str(i)] = [1.0 if x == i else 0.0 for x in new_df[e]]
                new_df[e + "-" + str(i)].astype('float')                                     #4
            new_df.drop(e, axis=1, inplace=True)                                                  #5
    return new_df

def accuracy(df, correctlabels):
    truCases, allCases = 0, 0
    for r in df.index:
        if correctlabels[r] == df.loc[r][:].idxmax(axis=0):
            truCases += df.loc[r][:].max(axis=0)
        allCases += df.loc[r][:].max(axis=0)
    return truCases/allCases

def brier_score(df, correctlabels):
    squared_sum = 0
    row = 0
    for label in correctlabels:
        i = np.where(df.columns==label)[0]
        for col in df.columns:
            squared_sum += (1 - df.loc[row, label])**2 if label==col else df.loc[row, col]**2
        row+=1
    return squared_sum/df.index.size

def auc(df, correctlabels):
    auc=0
    for col in df.columns:
        df2 = pd.concat([df[col], pd.Series(correctlabels.astype("category"), name='correct')], axis=1)
        # get dummies for correct labels and sort descending
        df2 = pd.get_dummies(df2.sort_values(col, ascending=False))
        
        # move col to first for easier total tp and fp calculation
        tmp=df2.pop('correct_' + str(col))
        # get the col frequency for calculating weighted AUCs
        col_frequency=tmp.sum()/tmp.index.size
        df2.insert(1, tmp.name, tmp)
#         display(df2)
        scores={}
        # populate the scores dictionary for column i.e. key=score, value=[tp_sum, fp_sum]
        for row in df.index:
            key=df2.iloc[row, 0]
#             current=np.zeros(score_dimension, dtype=np.uint) if scores.get(key) is None else scores[key]
            current=np.zeros(2, dtype=np.uint) if scores.get(key) is None else scores[key]
            to_add=np.array([1,0]) if df2.iloc[row, 1]==1 else np.array([0,1])
#             scores[key]=current+np.array([df2.iloc[row, 1], 0])
            scores[key]=current+to_add
#         print(scores)

        # calculate auc based on scores
#         print(f'scores={scores}')
        cov_tp=0
        column_auc=0
        tot_tp=0
        tot_fp=0
        # calculate total tp and fp 
        for value in scores.values():
            tot_tp+=int(value[0])
            tot_fp+=int(value[1])
            
        # same algorithm as in the lecture (bad naming though)
        for i in scores.values():
            if i[1] == 0:
                cov_tp+=i[0]
            elif i[0] == 0:
                column_auc += (cov_tp/tot_tp)*(i[1]/tot_fp)
            else:
                column_auc += (cov_tp/tot_tp)*(i[1]/tot_fp)+(i[0]/tot_tp)*(i[1]/tot_fp)/2
                cov_tp += i[0]
#         print(column_auc)    
        auc+=col_frequency*column_auc
    return auc
################### ASSIGNMENT 2 - kNN #################

""" represents an instace of k nearest neighbours
uses the function fit to train the learning algorithm
predict then uses the train data to predict certain class
label every for test instance 
"""
class kNN:
    def __init__(self):
      self.column_filter = None         # filter for columns with only missing values or 1 unique value
      self.imputation = None            # a mapping from colum to value to impute with
      self.normalization = None         # normalization from column to the max/min values to normalise with
      self.one_hot = None               # a mapping from a column to all possible categorical features
      self.labels = None                # all possible class labels from the training data
      self.training_labels = None       # a series containing class labels 
      self.training_data = None         # the training data used for predicting test instances
      self.training_time = None         # the time to train the algorithm
      self.df_Before = None             # USED FOR COMPARISON
      
    """takes a dataframe df, a normalization type (default=minmax), 
    prepare the current kNN instance for predictng test values
    """
    def fit(self, df, normalizationtype="minmax"): 
        df, self.column_filter = create_column_filter(df)
        df, self.imputation = create_imputation(df)
        df, self.normalization = create_normalization(df, normalizationtype)
        df, self.one_hot = create_one_hot(df)                                  # not needed
        self.training_labels = df["CLASS"].astype('category')
        self.labels = self.training_labels.cat.categories
        self.training_data = df.drop(["ID", "CLASS"], axis=1, errors="ignore").to_numpy()

    """takes a dataframe df, a constant k (default=5),
    returns a dataframe consisting of class labels as columns
    and rows as the probabilities for each class label for each
    row in df
    
    the probability of each label is estimated by the relative frequency in the set of labels
    from the k nearest (Euclidean distance) neighbors in training_data.
    """
    def predict(self, df, k=5):
        
        #1
        df_dropped = df.drop(["ID", "CLASS"], axis=1, errors="ignore")
        df_dropped = apply_column_filter(df_dropped, self.column_filter)
        df_dropped = apply_imputation(df_dropped, self.imputation)
        df_dropped = apply_normalization(df_dropped, self.normalization)
        #df_dropped = apply_one_hot(df_dropped, self.one_hot)                   # not needed
        
        #2
        predictions = pd.DataFrame(index=[], columns=self.labels)
        for i in df_dropped.index: 
            test_set = df_dropped.iloc[i,:].to_numpy()
            freq = self.get_nearest_neighbor_predictions(test_set, k)
            predictions = predictions.append(freq, ignore_index=True)
        return predictions
        
    def get_nearest_neighbor_predictions(self, x_test, k):
        distance_to_labels = sorted([(norm(x_test - x_train), i) for i, x_train in enumerate(self.training_data)])
        k_nearest = [self.training_labels[distance_to_labels[i][1]] for i in range(k)]
        class_freq = {c: k_nearest.count(c)/len(k_nearest) for c in self.labels}
        return class_freq
    
    
        
#############TESTING#############
        #predictions = pd.DataFrame(index=self.labels, columns=[get_nearest_neighbor_predictions(df_dropped.iloc[i,:].to_numpy(), k)for i in df_dropped.index])

glass_train_df = pd.read_csv("glass_train.csv")
glass_test_df = pd.read_csv("glass_test.csv")
temp = kNN()

#fit
t0 = time.perf_counter()
temp.fit(glass_train_df)
print("Training time: {0:.2f} s.".format(time.perf_counter()-t0))

#predict
test_labels = glass_test_df["CLASS"]
k_values = [1,3,5,7,9]
results = np.empty((len(k_values),3))

for i in range(len(k_values)):
    t0 = time.perf_counter()
    predictions = temp.predict(glass_test_df, k=k_values[i])
    print("Testing time (k={0}): {1:.2f} s.".format(k_values[i], time.perf_counter()-t0))
    results[i] = [accuracy(predictions,test_labels),brier_score(predictions,test_labels),
                  auc(predictions,test_labels)]

train_labels = glass_train_df["CLASS"]
predictions = temp.predict(glass_train_df,k=1)
print("Accuracy on training set (k=1): {0:.4f}".format(accuracy(predictions,train_labels)))

results = pd.DataFrame(results,index=k_values,columns=["Accuracy","Brier score","AUC"])

print("\nresults\n",results)
#display("results",results)