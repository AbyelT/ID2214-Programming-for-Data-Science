import numpy as np
import pandas as pd
import time

###################FUNCTIONS FROM ASSIGN 1##########################

def create_column_filter(df):
    new_df = df.copy()
    new_df = new_df.filter([e for e in new_df.columns if new_df[e].nunique() > 1 or e in ['CLASS', 'ID']], axis=1)
    return new_df, list(new_df.columns)

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

################### ASSIGNMENT 2 - Naive Bayes #################

class NaiveBayes:
    def __init__(self):
        self.column_filter = None                   # filter for columns with only missing values or 1 unique value
        self.binning = None                         # mapping of discrete intervals e.g (1-3), (3-5) etc.
        self.class_priors = None                    # mapping from class labels (from CLASS column) to the relative frequency of each label in the column
        self.labels = None                          # a list of the different categories from the CLASS column
        self.feature_class_value_counts = None      # mapping from feature to the number training instances with ...
        self.feature_class_counts = None            # mapping from feature to number training instances with ...

    def fit(self, df, nobins=10, bintype="equal-width"):
        # prepare data
        df, self.column_filter = create_column_filter(df)
        df, self.binning = create_bins(df, nobins, bintype)
        self.labels = df["CLASS"].astype('category').cat.categories
        self.class_priors = {c: sum(df["CLASS"] == c)/len(df["CLASS"]) for c in self.labels} # P(H)
        
        # for P(H|C)
        df_no_classId = df.drop(["ID", "CLASS"], axis=1)
        self.feature_class_value_counts = {(f, c, val): sum(df_no_classId[f] == val) for f in df_no_classId.columns for c in self.labels for val in pd.unique(df_no_classId[f])}
        self.feature_class_counts = {(f, c): sum(df["CLASS"] == c) for f in df_no_classId.columns for c in self.labels}



# Output from fit:
# <nothing>
#
# The result of applying this function should be:
#
# self.column_filter              - a column filter (see Assignment 1) from df
# self.binning                    - a discretization mapping (see Assignment 1) from df
# self.class_priors               - a mapping (dictionary) from the labels (categories) of the "CLASS" column of df,
#                                   to the relative frequencies of the labels
# self.labels                     - a list of the categories (class labels) of the "CLASS" column of df
# self.feature_class_value_counts - a mapping from the feature (column name) to the number of
#                                   training instances with a specific combination of (non-missing, categorical) 
#                                   value for the feature and class label
# self.feature_class_counts       - a mapping from the feature (column name) to the number of
#                                   training instances with a specific class label and some (non-missing, categorical) 
#                                   value for the feature
#
# Note that the function does not return anything but just assigns values to the attributes of the object.
#
# Input to predict:
# self - the object itself
# df   - a dataframe
# 
# Output from predict:
# predictions - a dataframe with class labels as column names and the rows corresponding to
#               predictions with estimated class probabilities for each row in df, where the class probabilities
#               are estimated by the naive approximation of Bayes rule (see lecture slides)
#
# Hint 1: First apply the column filter and discretization
#
# Hint 2: Iterating over either columns or rows, and for each possible class label, calculate the relative
#         frequency of the observed feature value given the class (using feature_class_value_counts and 
#         feature_class_counts) 
#
# Hint 3: Calculate the non-normalized estimated class probabilities by multiplying the class priors to the
#         product of the relative frequencies
#
# Hint 4: Normalize the probabilities by dividing by the sum of the non-normalized probabilities; in case
#         this sum is zero, then set the probabilities to the class priors

#############TESTING#############

glass_train_df = pd.read_csv("glass_train.csv")
glass_test_df = pd.read_csv("glass_test.csv")

nb_model = NaiveBayes()
test_labels = glass_test_df["CLASS"]

nobins_values = [3,5,10]
bintype_values = ["equal-width","equal-size"]
parameters = [(nobins,bintype) for nobins in nobins_values for bintype in bintype_values]

results = np.empty((len(parameters),3))

for i in range(len(parameters)):
    nb_model.fit(glass_train_df,nobins=parameters[i][0],bintype=parameters[i][1])
    print("works " + str(i))
