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

# ------------------ASSIGNMENT 2b------------------

class RandomForest:
    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.one_hot = None
        self.labels = None
        self.model = None
        self.oob_acc = 0.0

    def fit(self, df, no_trees=100):
        new_df = df.copy()
        new_df, self.column_filter = create_column_filter(new_df)
        new_df, self.imputation = create_imputation(new_df)
        new_df, self.one_hot = create_one_hot(new_df)

        self.labels = np.array(new_df["CLASS"].values)         
        new_df.drop(["CLASS"], axis=1, inplace=True)
        instances = np.array(new_df.values)

        self.model = []
        rng = np.random.default_rng()
        
        ## task 2b
        c_labels = np.unique(sorted(self.labels))   #1 for 2b
        mapping = {instance: i for i, instance in enumerate(c_labels)}
        oob_predictions = pd.DataFrame(0, index=new_df.index, columns=c_labels)  
        oob_vector = pd.Series(0, index=new_df.index)
        
        for i in range(no_trees):
            features = int(log2(len(new_df.columns)))
            tree = DecisionTreeClassifier(max_features=features)                                              # 1. generate decision tree       #int(log2(len(df.columns))
            bootstrap_indexes = rng.choice(np.array(new_df.index), size=len(new_df.index), replace=True)    # 2. create list of random indexes for random sampling with replacement
            bootstrap_sample = [instances[e] for e in bootstrap_indexes]                                   # 3. create single bootstrap sample for model
            bootstrap_labels = [self.labels[e] for e in bootstrap_indexes]
            self.model.append(tree.fit(bootstrap_sample, bootstrap_labels))    
                               # 4. train base model with bootstrap instances
            ## task 2b: for oob predictions
            bootstrap_sample = np.array(bootstrap_sample)
            notIncluded_instances = []
            for indx in new_df.index: 
                if indx not in bootstrap_indexes:
                    #print(str(indx) + " not in bootstrap indexes!")
                    current_instance = np.array(instances[indx]).reshape(1, -1)
                    X = tree.predict_proba(current_instance)
                    for col in range(X.shape[1]):
                         current_label = tree.classes_[col]
                         correct_col = mapping.get(current_label)
                         oob_predictions.iloc[indx,correct_col] += X[0][col]   
                         oob_vector[indx] += 1
                    continue
        print("tree done")
        
        oob_predictions = oob_predictions.div(oob_vector, axis=0)
        self.oob_acc = accuracy(oob_predictions, df["CLASS"])
        
# print(bootstrap_sample.isin(instances[indx]))
# print(instances[indx])
# print(instances[indx] in bootstrap_sample)
#if bootstrap_sample[indx] is in instances:
# current_instance = np.array(instances[indx]).reshape(1, -1)
# if current_instance not in bootstrap_sample:
#     notIncluded_instances.append(instances[indx])
# tmp = np.array(instances[indx]).reshape(1, -1)
# X = tree.predict_proba(tmp)
# for col in range(X.shape[1]):
#     current_label = tree.classes_[col]
#     correct_col = mapping.get(current_label)
#     oob_predictions.iloc[indx,correct_col] += X[0][col]   
#     oob_vector[indx] += 1
            
    def predict(self, df):

        new_df = df.copy()
        new_df.drop(["CLASS"], axis=1, inplace=True)
        new_df = apply_column_filter(new_df, self.column_filter)
        new_df = apply_imputation(new_df, self.imputation)
        new_df = apply_one_hot(new_df, self.one_hot)
        
        class_labels = np.unique(sorted(self.labels))
        mapping = {instance: i for i, instance in enumerate(class_labels)}

        y_predictions = pd.DataFrame(0, index=df.index, columns=class_labels)  
        for i, tree in enumerate(self.model):
            prediction_i = tree.predict_proba(new_df.values)
            for col in range(prediction_i.shape[1]):
                current_label = tree.classes_[col]
                correct_col = mapping.get(current_label)
                y_predictions.iloc[:,correct_col] += prediction_i[:, col]      
        pred = y_predictions/len(self.model)
        return pred
    
# Define an extended version of the class RandomForest with the same input and output as described in part 2a above,
# where the results of the fit function also should include:
# self.oob_acc - the accuracy estimated on the out-of-bag predictions, i.e., the fraction of training instances for 
#                which the given (correct) label is the same as the predicted label when using only trees for which
#                the instance is out-of-bag
#
# Hint 1: You may first create a zero matrix with one row for each training instance and one column for each class label
#         and one zero vector to allow for storing aggregated out-of-bag predictions and the number of out-of-bag predictions
#         for each training instance, respectively
#
# Hint 2: After generating a tree in the forest, iterate over the indexes that were not included in the bootstrap sample
#         and add a prediction of the tree to the out-of-bag prediction matrix and update the count vector
#
# Hint 3: Note that the input to predict_proba has to be a matrix; from a single vector (row) x, a matrix with one row
#         can be obtained by x[None,:]
#
# Hint 4: Finally, divide each row in the out-of-bag prediction matrix with the corresponding element of the count vector

# Test your code (leave this part unchanged, except for if auc is undefined)

train_df = pd.read_csv("anneal_train.csv")

test_df = pd.read_csv("anneal_test.csv")

rf = RandomForest()

t0 = time.perf_counter()
rf.fit(train_df)
print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

print("OOB accuracy: {:.4f}".format(rf.oob_acc))

test_labels = test_df["CLASS"]

t0 = time.perf_counter()
predictions = rf.predict(test_df)
print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))

print("Accuracy: {:.4f}".format(accuracy(predictions,test_labels)))
print("AUC: {:.4f}".format(auc(predictions,test_labels))) # Comment this out if not implemented in assignment 1
print("Brier score: {:.4f}".format(brier_score(predictions,test_labels))) # Comment this out if not implemented in assignment 1
