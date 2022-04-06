import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display

def create_column_filter(df):
    df2 = df.copy()
    column_filter = list(df2.columns)
    columns = [col for col in df2.columns if col not in ['active', 'index', 'id', 'class']]
    for col in columns:
        if df2[col].isnull().all():
            df2.drop(columns=col, inplace=True)
            column_filter.remove(col)
            continue

        if len(df2[col].dropna().unique()) <= 1:
            df2.drop(columns=col, inplace=True)
            column_filter.remove(col)

    return df2, column_filter


def apply_column_filter(df, column_filter):
    df2 = df.copy()
    [df2.drop(columns=col, inplace=True) for col in df2.columns if col not in column_filter]
    return df2

def create_normalization(df, normalizationtype='minmax'):
    df2 = df.copy()
    include_types = np.int32, np.int64, np.float32, np.float64
    columns = [col for col in df2.columns if col not in ['active', 'index', 'id', 'class']
               and df2[col].dtype in include_types]
    normalization = {}
    for col in columns:
        if normalizationtype=='minmax':
            min = df2[col].min()
            max = df2[col].max()
            normalization[col] = normalizationtype, min, max
        elif normalization=='zscore':
            mean = df2[col].mean()
            std = df[col].std()
            normalization = normalizationtype, mean, std

    for col in columns:
        values = list(normalization[col])
        if values[0] == 'minmax':
            df2[col] = [(x-values[1])/(values[2]-values[1]) for x in df[col]]

    return df2, normalization

def apply_normalization(df, normalization):
    df2 = df.copy()
    include_types = np.int32, np.int64, np.float32, np.float64
    columns = [col for col in df2.columns if col not in ['active', 'index', 'id', 'class']
               and df2[col].dtype in include_types]
    for col in columns:
        values = list(normalization[col])
        if values[0] == 'minmax':
            df2[col] = [(x-values[1])/(values[2]-values[1]) for x in df[col]]


    return df2

def create_imputation(df):
    df2 = df.copy()
    numeric_types = np.int32, np.int64, np.float32, np.float64
    columns = [col for col in df2.columns if col not in ['active', 'index', 'id', 'class']]
    imputation = {}
    for col in columns:
        if df2[col].dtype in numeric_types:
            if df2[col].isnull().all():
                df2[col].fillna(0, inplace=True)
            imputation[col] = df2[col].mean()
            df2[col].fillna(df2[col].mean(), inplace=True)
        else:
            if df2[col].isnull().all():
                df2[col].fillna('', inplace=True) if df2[col].dtype == 'object' else \
                    df2[col].astype('category') and df2[col].fillna(df2[col].cat.categories[0], inplace=True)

            imputation[col] = df2[col].mode()[0]
            df2[col].fillna(imputation[col], inplace=True)

    return df2, imputation

def apply_imputation(df, imputation):
    df2 = df.copy()
    return df2.fillna(value=imputation)

def create_one_hot(df):
    df2=df.copy()
    columns = [col for col in df2.columns if col not in ['active', 'index', 'id', 'class']]
    one_hot={}
    for col in columns:
        if df2[col].dtype.name != 'category' and df2[col].dtype.name != 'object':
            continue
        one_hot[col]=df2[col].unique()
        tmp = pd.get_dummies(df2[col], prefix=col, prefix_sep='-', dtype=np.float64)
        df2.drop(columns=col, inplace=True)
        df2 = pd.concat([df2, tmp], axis=1)

    return df2, one_hot

def apply_one_hot(df, one_hot):
    new_df = df.copy()
    display(one_hot)
    for e in new_df.columns:
        if e in one_hot:
            for i in one_hot[e]: 
                new_df[e + "-" + i] = [1.0 if x == i else 0.0 for x in new_df[e]]
                new_df[e + "-" + i].astype('float')                                     #4
            new_df = new_df.drop(e, axis=1)                                             #5
    return new_df

def accuracy(df, correctlabels):
    highest_probability = df.idxmax(axis=1)
    correct_occurances = 0
    for correct_label, predicted_label in zip(correctlabels, highest_probability):
        if correct_label==predicted_label:
            correct_occurances+=1

    return correct_occurances/df.index.size

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
        df2 = pd.concat([df[col], pd.Series(correctlabels.astype('category'), name='correct')], axis=1)
        # get dummies for correct labels and sort descending
        df2 = pd.get_dummies(df2.sort_values(col, ascending=False))

        # move col to first for easier total tp and fp calculation
        tmp=df2.pop('correct_'+str(col))
        # get the col frequency for calculating weighted AUCs
        col_frequency=tmp.sum()/tmp.index.size
        df2.insert(1, tmp.name, tmp)
        scores={}
        # populate the scores dictionary for column i.e. key=score, value=[tp_sum, fp_sum]
        for row in df.index:
            key=df2.iloc[row, 0]
            current=np.zeros(2, dtype=np.uint) if scores.get(key) is None else scores[key]
            to_add=np.array([1,0]) if df2.iloc[row, 1]==1 else np.array([0,1])
            scores[key]=current+to_add

        # calculate auc based on scores
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

        auc+=col_frequency*column_auc

    return auc

def create_bins(df, nobins=10, bintype='equal-width'):
    df2=df.copy()
    include_types = np.int32, np.int64, np.float32, np.float64
    columns = [col for col in df2.columns if (col != 'active' and col != 'index')
               and df2[col].dtype in include_types]
    binning ={}
    for col in columns:
        df2[col].astype('category')
        if (bintype == 'equal-width'):
            res, bins = pd.cut(df2[col], nobins, retbins=True, labels=False)
            bins[0] = -np.inf
            bins[len(bins)-1] = np.inf
            binning[col] = bins
        elif (bintype == 'equal-size'):
            # We drop duplicates which results in fewer bins
            res, bins = pd.qcut(df2[col], nobins, retbins=True, labels=False, duplicates='drop')
            bins[0] = -np.inf
            bins[len(bins)-1] = np.inf
            binning[col] = bins
    return df2, binning

def apply_bins(df, binning):
    df2 = df.copy()
    columns = [col for col in df2.columns if (col != 'active' and col != 'index')]
    for col in columns:
        labels=list(range(len(binning[col])-1))
        df2[col] = pd.cut(df2[col], binning[col], labels=labels)
    return df2

def split(df,testfraction=0.5):
    df2= df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
    cutoff = int(testfraction*len(df))
    testdf = df2.iloc[:cutoff,:]
    trainingdf = df2.iloc[cutoff:,:]

    testdf = pd.DataFrame(testdf, columns=list(df.columns))
    trainingdf = pd.DataFrame(trainingdf, columns=list(df.columns))

    #print('TEST1',trainingdf.dtypes)

    return trainingdf, testdf

class RandomForest:
    def __init__(self):
        self.column_filter = None
        self.imputation = None
        #self.one_hot = None
        self.labels = None
        self.model = None
        self.oob_acc = None

    def fit(self, df, no_trees=100):
        df2 = df.copy()
        #print('PROVA', len(df2))
        #display(df)
        df2, self.column_filter = create_column_filter(df2)
        df2, self.imputation = create_imputation(df2)
        #df2, self.one_hot = create_one_hot(df2)

        # Set labels
        self.labels = np.array(pd.Series(df2['active'].astype('category').unique().sort_values()))

        training_labels = df2['active'].values
        df2.drop(['active','index'], axis='columns', inplace=True)
        feature_names = df2.columns
        #display(df2)

        # Type casting to int uses floor value. With ceiling value, accuracy improves
        #max_features = int(np.log2(df2.columns.size))
        self.model = []
        oob_predictions = pd.DataFrame(np.zeros((df2.index.size, self.labels.size)), columns=self.labels)
        oob_predictions_count = np.zeros(df2.index.size)
        training_labels = df["active"]

        dt = DecisionTreeClassifier(max_features='log2')

        for i in range(no_trees):
            bootstrap_instances = np.random.choice(df2.shape[0], df2.shape[0], replace=True)
            #print('TEST333', len(bootstrap_instances))
            #X = df2.iloc[bootstrap_instances, :].values
            X = [df2.values[i] for i in bootstrap_instances]
            y = [df['active'].values[i] for i in bootstrap_instances]
            #y = training_labels[bootstrap_instances]
            #tree = DecisionTreeClassifier(max_features='log2')
            tree = dt.fit(X,y)
            #tree.fit(X, y)
            self.model.append(tree)

            # Set out-of-bag predictions

            oob_indexes = df2.index.difference(bootstrap_instances)
            #print('TEST_444', len(oob_indexes))

            #for j in oob_indexes:
            #    if j>len(df2): print('BINGO')

            #tmp = tree.predict_proba(df2.iloc[oob_indexes, :].values)
            tmp = tree.predict_proba([df2.values[i] for i in oob_indexes])

            oob_predictions_count[oob_indexes] += 1
            for label, j in zip(tree.classes_, range(len(tree.classes_))):
                label_index = np.where(self.labels == label)[0][0]
                oob_predictions.loc[oob_indexes, self.labels[label_index]] += tmp[:, j]


        for i in range(oob_predictions.index.size):
            oob_predictions.iloc[i, :] /= oob_predictions_count[i]

        self.oob_acc = accuracy(oob_predictions, training_labels)

    def predict(self, df):
        df2 = df.copy()
        df2.drop(columns=['active','index'], inplace=True)
        df2 = apply_column_filter(df2, self.column_filter)
        df2 = apply_imputation(df2, self.imputation)
        #df2 = apply_one_hot(df2, self.one_hot)
        predictions = pd.DataFrame(np.zeros((df2.index.size, self.labels.size)), columns=self.labels)

        for tree in self.model:
            tree_predictions = tree.predict_proba(df2.values)
            for label, i in zip(tree.classes_, range(len(tree.classes_))):
                label_index = np.where(self.labels == label)[0][0]
                predictions[self.labels[label_index]] += tree_predictions[:, i]

        return predictions / len(self.model)


class kNN:
    def __init__(self):
        self.column_filter = None
        self.imputation = None
        self.normalization = None
        #self.one_hot = None
        self.labels = None
        self.training_labels = None
        self.training_data = None
        self.training_time = None

    def fit(self, df, normalizationtype='minmax'):
        df2 = df.copy()
        df2.drop(columns=['active', 'index'], inplace=True)
        df2, self.column_filter = create_column_filter(df2)
        df2, self.imputation = create_imputation(df2)
        df2, self.normalization = create_normalization(df2, normalizationtype)
        #df2, self.one_hot = create_one_hot(df2)

        # Set training labels
        self.training_labels = pd.Series(df['active'].astype('category'))

        # Set labels
        self.labels = np.array(self.training_labels.unique().sort_values())

        # Set training_data
        self.training_data = df2.values

    def get_nearest_neighbor_predictions(self, x_test, k):
        # key=index value=distance series
        distances = {}
        for index, row in enumerate(self.training_data):
            distance = np.sqrt(np.sum( (row - x_test)**2 ))
            distances[index] = distance

        distances = sorted(distances.items(), key=lambda x:x[1])
        k_nearest_labels = np.array([self.training_labels[distances[i][0]] for i in range(k)])
        return k_nearest_labels


    def predict(self, df, k=5):
        df2 = df.copy()
        df2.drop(columns=['active', 'index'], inplace=True)
        df2 = apply_column_filter(df2, self.column_filter)
        df2 = apply_imputation(df2, self.imputation)
        df2 = apply_normalization(df2, self.normalization)
        #df2 = apply_one_hot(df2, None)
        predictions = pd.DataFrame(np.zeros((df2.index.size, self.labels.size)), columns=self.labels)
        for index, row in df2.iterrows():
            raw_k_neighbours = self.get_nearest_neighbor_predictions(df2.iloc[index,:].values, k)
            # get probabilities
            for col in self.labels:
                if col not in raw_k_neighbours:
                    continue
                else:
                    predictions.loc[index, col] = np.count_nonzero(raw_k_neighbours==col)/raw_k_neighbours.size

        return predictions

class NaiveBayes:
    def __init__(self):
        # column_filter, binning, labels, class_priors, feature_class_value_counts, feature_class_counts
        self.column_filter = None
        self.binning = None
        self.labels = None
        self.class_priors = None
        self.feature_class_value_counts = None
        self.feature_class_counts = None

    def fit(self, df, nobins=10, bintype='equal-width'):
        df2 = df.copy()
        df2, self.column_filter = create_column_filter(df2)
        df2, self.binning = create_bins(df2, nobins, bintype)

        # Set labels
        self.labels = np.array(pd.Series(df['active'].astype('category').sort_values()).unique())

        # Set class_priors = P(H)
        self.class_priors = {}
        for label in self.labels:
            prior = np.count_nonzero(df2['active'] == label) / df2['active'].size
            self.class_priors[label] = prior

        # Set class_value_counts = numerator in P(Xi|H) for i in {features}
        self.feature_class_value_counts = {}
        # Set class_counts = denominator in P(Xi|H) for i in {features}
        self.feature_class_counts = {}

        columns = [col for col in df2.columns if (col != 'active' and col != 'index')]

        # Class labels are sorted so we son't add them assuming index maps to the correct label later
        # Assumption: NANs are dropped or purned in data preparation step
        class_counts = []
        for label in self.labels:
            class_counts += [np.count_nonzero(df2['active'] == label)]
        for col in columns:
            self.feature_class_counts[col] = class_counts
            self.feature_class_value_counts[col] = []

        for index, label in enumerate(self.labels):
            for col in columns:
                value_count = {}
                for unique in df2[col].unique():
                    value_count[unique] = 0
                for value, clazz in zip(df2[col], df2['active']):
                    if clazz != label:
                        continue
                    else:
                        value_count[value] += 1
                self.feature_class_value_counts[col].append(value_count)

    def predict(self, df):
        df2 = df.copy()
        df2 = apply_column_filter(df2, self.column_filter)
        df2 = apply_bins(df2, self.binning)
        predictions = pd.DataFrame(np.zeros((df2.index.size, self.labels.size)), columns=self.labels)
        columns = [col for col in df2.columns if (col != 'active' and col != 'index')]

        # Calculate non-normalized probabilities
        for label_index, label in enumerate(self.labels):
            for row_index, row in df2.iterrows():
                relative_frequency = 1
                # sum = 0
                for col in columns:
                    value = row[col]
                    label_feature_value_count = self.feature_class_value_counts[col][label_index]
                    try:
                        numerator = label_feature_value_count[value]
                    except KeyError:
                        continue
                    denominator = self.feature_class_counts[col][label_index]
                    relative_frequency *= (numerator / denominator)
                probability = relative_frequency * self.class_priors[label]
                predictions.iloc[row_index, label_index] = probability

       # Normalize
        for index, row in predictions.iterrows():
            sum = 0
            for col in self.labels:
                sum += row[col]
            if sum == 0:
                sum = self.class_priors[col]
            for col in self.labels:
                predictions.loc[index, col] /=sum

        return predictions
