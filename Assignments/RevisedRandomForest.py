import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from math import log2

# ------------------FUNCTIONS FROM ASSIGN 1------------------


def create_column_filter(df):
    new_df = df.copy()
    new_df = new_df.filter([e for e in new_df.columns if new_df[e].nunique() > 1 or e in ['CLASS', 'ID']], axis=1)
    return new_df, list(new_df.columns)

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

def apply_normalization(df, normalization):
    new_df = df.copy()
    for e in new_df.columns:
        if e in normalization:
            a = normalization.get(e)[1]
            b = normalization.get(e)[2]
            new_df[e] = [(x-a)/(b-a) for x in new_df[e]] # works with minmax or zscore
    return new_df

def create_one_hot(df):
    new_df = df.copy()
    handle = new_df.filter([e for e in new_df.columns if e not in ['CLASS', 'ID']], axis=1)
    one_hot = {}
    for e in handle.columns:
        #print(new_df[e].dtypes.name)
        if new_df[e].dtypes.name == 'category':
            features = np.sort(handle[e].unique())  # 3
            for i in features:
                new_df[e + "-" + str(i)] = [1.0 if x == i else 0.0 for x in handle[e]]
                new_df[e + "-" + str(i)].astype('float')  # 4
            one_hot[e] = features
            new_df.drop(e, axis=1, inplace=True)  # 5
    return new_df, one_hot


def apply_column_filter(df, column_filter):
    new_df = df.copy()
    new_df = new_df.filter(items=column_filter, axis=1)
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

# ------------------ASSIGNMENT 2------------------

class RevisedForest:
    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.one_hot = None
        self.labels = None
        self.model = None

    def fit(self, df, no_trees=100):
        # 1
        new_df = df.copy()
        new_df, self.column_filter = create_column_filter(new_df)
        new_df, self.imputation = create_imputation(new_df)
        new_df, self.one_hot = create_one_hot(new_df)

        # 2
        self.labels = np.array((new_df["CLASS"].values))          
        new_df.drop(["CLASS"], axis=1, inplace=True)
        instances = np.array(new_df.values)

        # 3
        self.model = []
        rng = np.random.default_rng()
        for i in range(no_trees):
            features = int(log2(len(new_df.columns)))
            dt = DecisionTreeClassifier(max_features=features)                                              # 1. generate decision tree       #int(log2(len(df.columns))
            bootstrap_indexes = rng.choice(np.array(new_df.index), size=len(new_df.index), replace=True)    # 2. create list of random indexes for random sampling with replacement
            bootstrap_sample = [instances[e] for e in bootstrap_indexes]                                    # 3. create single bootstrap sample for model
            bootstrap_labels = [self.labels[e] for e in bootstrap_indexes]
            self.model.append(dt.fit(bootstrap_sample, bootstrap_labels))                                   # 4. train base model with bootstrap instances
        
    def predict(self, df):

        # 1
        new_df = df.copy()
        new_df.drop(["CLASS"], axis=1, inplace=True)
        new_df = apply_column_filter(new_df, self.column_filter)
        new_df = apply_imputation(new_df, self.imputation)
        new_df = apply_one_hot(new_df, self.one_hot)
        
        class_labels = np.unique(sorted(self.labels))
        mapping = {instance: i for i, instance in enumerate(class_labels)}

        # 2
        y_predictions = pd.DataFrame(0, index=df.index, columns=class_labels)  
        for i, tree in enumerate(self.model):
            prediction_i = tree.predict_proba(new_df.values)
            for col in range(prediction_i.shape[1]):
                current_label = tree.classes_[col]
                correct_col = mapping.get(current_label)
                y_predictions.iloc[:,correct_col] += prediction_i[:, col]      
        pred = y_predictions/len(self.model)
        return pred
    
# Hint 1: The categories obtained with <pandas series>.cat.categories are sorted in the same way as the class labels
#         of a DecisionTreeClassifier; the latter are obtained by <DecisionTreeClassifier>.classes_ 
#         The problem is that classes_ may not include all possible labels, and hence the individual predictions 
#         obtained by <DecisionTreeClassifier>.predict_proba may be of different length or even if they are of the same
#         length do not necessarily refer to the same class labels. You may assume that each class label that is not included
#         in a bootstrap sample should be assigned zero probability by the tree generated from the bootstrap sample.   
#
# Hint 2: Create a mapping from the complete (and sorted) set of class labels l0, ..., lk-1 to a set of indexes 0, ..., k-1,
#         where k is the number of classes
#
# Hint 3: For each tree t in the forest, create a (zero) matrix with one row per test instance and one column per class label,
#         to which one column is added at a time from the output of t.predict_proba 
#
# Hint 4: For each column output by t.predict_proba, its index i may be used to obtain its label by t.classes_[i];
#         you may then obtain the index of this label in the ordered list of all possible labels from the above mapping (hint 2); 
#         this index points to which column in the prediction matrix the output column should be added to 
#    
# Test your code (leave this part unchanged, except for if auc is undefined)

train_df = pd.read_csv("anneal_train.csv")

test_df = pd.read_csv("anneal_test.csv")

rf = RevisedForest()

t0 = time.perf_counter()
rf.fit(train_df)
print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

test_labels = test_df["CLASS"]

t0 = time.perf_counter()
predictions = rf.predict(test_df)
print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))

print("Accuracy: {:.4f}".format(accuracy(predictions,test_labels)))
print("AUC: {:.4f}".format(auc(predictions,test_labels))) # Comment this out if not implemented in assignment 1
print("Brier score: {:.4f}".format(brier_score(predictions,test_labels))) # Comment this out if not implemented in assignment 1