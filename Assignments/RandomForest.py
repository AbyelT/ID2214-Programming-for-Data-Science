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
                    new_df[e].astype('category')
                if new_df[e].nunique() < 1:
                    new_df[e].fillna(new_df[e][0], inplace=True)
                new_df[e].fillna(new_df[e].mode()[0], inplace=True)
                imputation[e] = new_df[e].mode()[0]
    return new_df, imputation


def create_one_hot(df):
    new_df = df.copy()
    handle = new_df.filter([e for e in new_df.columns if new_df[e].dtype.name in [
                           'object', 'category'] and e not in ['CLASS', 'ID']], axis=1)
    one_hot = {}
    for e in handle.columns:
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

class RandomForest:
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
        # .astype('category').cat.categories.to_numpy()
        self.labels = np.array(new_df["CLASS"].values)
        new_df.drop(["CLASS"], axis=1, inplace=True)
        instances = np.array(new_df.values)

        # 3
        self.model = []
        rng = np.random.default_rng()
        for i in range(no_trees):
            features = int(log2(len(new_df.columns)))
            dt = DecisionTreeClassifier(max_features=features)                                     # 1. generate decision tree       #int(log2(len(df.columns))
            bootstrap_indexes = rng.choice(np.array(new_df.index), size=len(new_df.index), replace=True) # 2. create list of random indexes for random sampling with replacement
            bootstrap_sample = [instances[e] for e in bootstrap_indexes]                         # 3. create single bootstrap sample for model
            bootstrap_labels = [self.labels[e] for e in bootstrap_indexes]
            self.model.append(dt.fit(bootstrap_sample, bootstrap_labels))                             # 4. train base model with bootstrap instances

    def predict(self, df):

        # 1
        new_df = df.copy()
        new_df.drop(["CLASS"], axis=1, inplace=True)
        new_df = apply_column_filter(new_df, self.column_filter)
        new_df = apply_imputation(new_df, self.imputation)
        new_df = apply_one_hot(new_df, self.one_hot)

        # 2
        y_predictions = pd.DataFrame(0, index=df.index, columns=np.unique(self.labels))  # before: np.unique(self.labels)
        for i, tree in enumerate(self.model):
            prediction_i = tree.predict_proba(new_df.values)
            y_predictions = y_predictions.add(prediction_i)
        pred = y_predictions/len(self.model)
        return pred
    


# Output from predict:
# predictions - a dataframe with class labels as column names and the rows corresponding to
#               predictions with estimated class probabilities for each row in df, where the class probabilities
#               are the averaged probabilities output by each decision tree in the forest
#
# Hint 1: Drop any "CLASS" and "ID" columns of the dataframe first and then apply column filter, imputation and one_hot ok
#
# Hint 2: Iterate over the trees in the forest to get the prediction of each tree by the method predict_proba(X) where
#         X are the (numerical) values of the transformed dataframe; you may get the average predictions of all trees,
#         by first creating a zero-matrix with one row for each test instance and one column for each class label,
#         to which you add the prediction of each tree on each iteration, and then finally divide the prediction matrix
#         by the number of trees.
#
# Hint 3: You may assume that each bootstrap sample that was used to generate each tree has included all possible
#         class labels and hence the prediction of each tree will contain probabilities for all class labels
#         (in the same order). Note that this assumption may be violated, and this limitation will be addressed
#         in the next part of the assignment.

train_df = pd.read_csv("tic-tac-toe_train.csv")
test_df = pd.read_csv("tic-tac-toe_test.csv")

rf = RandomForest()

t0 = time.perf_counter()
rf.fit(train_df)
print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

test_labels = test_df["CLASS"]
t0 = time.perf_counter()
predictions = rf.predict(test_df)

print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))
print("Accuracy: {:.4f}".format(accuracy(predictions, test_labels)))
# Comment this out if not implemented in assignment 1
print("AUC: {:.4f}".format(auc(predictions, test_labels)))
# Comment this out if not implemented in assignment 1
print("Brier score: {:.4f}".format(brier_score(predictions, test_labels)))

# train_labels = train_df["CLASS"]
# predictions = rf.predict(train_df)
# print("Accuracy on training set: {0:.4f}".format(accuracy(predictions,train_labels)))
# print("AUC on training set: {0:.4f}".format(auc(predictions,train_labels))) # Comment this out if not implemented in assignment 1
# print("Brier score on training set: {0:.4f}".format(brier_score(predictions,train_labels))) # Comment this out if not implemented in assignment 1
